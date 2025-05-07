# -*- coding: utf-8 -*-
"""
Integrated UA-SLSM Tissue Preprocessing Pipeline.

This script combines functionality from three separate scripts:
1. rename_images.py - Renames images and extracts metadata
2. normalize_and_stack.py - Performs exposure normalization and stacks to 3 channels
3. run.py - Calculates entropy, depth maps, and other tissue metrics

Provides a unified interface for the entire preprocessing workflow.
"""

import sys
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import re
import os
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, List, Union

# --- Third-party libraries ---
import cv2  # OpenCV for image reading/writing
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data handling
from scipy.ndimage import (  # SciPy for image processing tasks
    gaussian_filter,
    median_filter,
    distance_transform_edt,
    label,
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    uniform_filter1d  # For contour smoothing
)
from skimage.filters.rank import entropy as skimage_entropy  # scikit-image for entropy
from skimage.morphology import disk  # scikit-image for morphological elements
from skimage.util import img_as_ubyte, img_as_float32  # scikit-image utilities
from skimage.filters import gaussian  # scikit-image for Gaussian filter
from skimage.color import rgb2gray  # scikit-image for color conversion
from skimage.measure import regionprops  # scikit-image for region properties
from skimage.exposure import equalize_adapthist  # For AHE in entropy calculation

# --- Performance improvement for skimage ---
try:
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']
except Exception:
    pass

# --- Matplotlib Setup (Optional Dependency) ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    class plt:
        @staticmethod
        def subplots(*args, **kwargs): raise ImportError("Matplotlib not installed")
        @staticmethod
        def savefig(*args, **kwargs): raise ImportError("Matplotlib not installed")
        @staticmethod
        def close(*args, **kwargs): pass
        @staticmethod
        def tight_layout(*args, **kwargs): pass
        @staticmethod
        def colorbar(*args, **kwargs): pass
        @staticmethod
        def switch_backend(*args, **kwargs): pass
    class make_axes_locatable:
        def __init__(self, *args, **kwargs): raise ImportError("Matplotlib not installed")

# --- Configuration Defaults ---
# Image file formats
SUPPORTED_FORMATS = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
OUTPUT_RGB_EXTENSION = ".png"  # Output format for RGB/stacked images
OUTPUT_CHANNEL_EXTENSION = ".png"  # Output format for channel data

# Regular expression patterns for filename metadata extraction
FILENAME_PATTERN = r'Study(LSM\d+)Biopsy(\d+\w?)(?:a)?StepSize(\d+\.\d+)ExpoTime(\d+\.\d+)ms\((\d+)\)'
COMMON_SUFFIXES = ['_rgb_invD.png', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
FILE_EXTENSIONS = ['.tif', '.tiff', '.tiff.tif', '.png', '_rgb_invD.png', '.jpg', '.jpeg']

# Processing parameters
DEFAULT_ENTROPY_DISK_RADIUS = 3
DEFAULT_ENTROPY_GAMMA = 10
DEFAULT_ENTROPY_SIGMA = 0
DEFAULT_ENTROPY_MEDIAN_SIZE = 0
DEFAULT_ENTROPY_LOWER_PERCENTILE = 0
DEFAULT_ENTROPY_UPPER_PERCENTILE = 100
DEFAULT_USE_AHE = False
DEFAULT_AHE_CLIP_LIMIT = 0.001
DEFAULT_MAX_DEPTH_MAP = 340
DEFAULT_EXPOSURE_TIME = 0.1
DEFAULT_CONTOUR_SMOOTHING_WINDOW = 50

# Output channel suffixes
STATS_CHANNEL_SUFFIXES = {
    "normalized": "_normalized.png",
    "entropy": "_entropy.png",
    "depth": "_inv_depthmap.png"
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger('integrated_processor')


# =============================================================================
# == UTILITY FUNCTIONS
# =============================================================================

# Global variable to store metadata from CSV file
METADATA_DF = None

def load_metadata_csv(metadata_csv_path: str) -> Optional[pd.DataFrame]:
    """Load metadata from CSV file into a global DataFrame."""
    global METADATA_DF
    if METADATA_DF is None and metadata_csv_path and os.path.exists(metadata_csv_path):
        try:
            METADATA_DF = pd.read_csv(metadata_csv_path)
            logger.info(f"Loaded metadata from {metadata_csv_path} with {len(METADATA_DF)} entries")
            return METADATA_DF
        except Exception as e:
            logger.error(f"Error loading metadata CSV: {e}")
    return METADATA_DF

def extract_exposure_time(filename: str, metadata_csv_path: str = None) -> Optional[float]:
    """Extract exposure time from metadata.csv or filename.
    
    Args:
        filename: Image filename to extract exposure time from
        metadata_csv_path: Path to metadata CSV file (optional)
        
    Returns:
        Exposure time as float if found, None otherwise
    """
    # Try to get exposure time from metadata CSV first
    global METADATA_DF
    
    # Load metadata CSV if provided and not already loaded
    if metadata_csv_path:
        load_metadata_csv(metadata_csv_path)
    
    # Extract base filename without path and extension
    base_filename = os.path.basename(filename)
    base_name, _ = os.path.splitext(base_filename)
    
    # Try to find the exposure time in the metadata DataFrame
    if METADATA_DF is not None:
        # Try exact match on image_path column
        match = METADATA_DF[METADATA_DF['image_path'] == base_filename]
        
        # If no exact match, try matching on the base filename without extension
        if len(match) == 0:
            for ext in COMMON_SUFFIXES:
                potential_match = METADATA_DF[METADATA_DF['image_path'].str.startswith(base_name)]
                if len(potential_match) > 0:
                    match = potential_match
                    break
        
        # If we found a match, return the exposure time
        if len(match) > 0:
            try:
                expo_time = float(match.iloc[0]['expo_time'])
                logger.debug(f"Found exposure time {expo_time} for {base_filename} in metadata CSV")
                return expo_time
            except (ValueError, KeyError, IndexError) as e:
                logger.debug(f"Error extracting exposure time from metadata CSV: {e}")
    
    # Fall back to extracting from filename using regex
    match = re.search(r'ExpoTime([0-9.]+)', filename)
    if match:
        try: 
            expo_time = float(match.group(1))
            logger.debug(f"Extracted exposure time {expo_time} from filename {base_filename}")
            return expo_time
        except ValueError: 
            logger.debug(f"Failed to convert exposure time from filename {base_filename}")
            return None
    
    logger.debug(f"No exposure time found for {base_filename}")
    return None


def clean_image_name(image_name: str) -> str:
    """
    Clean image name by removing common suffixes.
    
    Args:
        image_name: Original image name
        
    Returns:
        String with common suffixes removed
    """
    # Remove each suffix if present
    cleaned_name = image_name
    for suffix in COMMON_SUFFIXES:
        if cleaned_name.lower().endswith(suffix.lower()):
            cleaned_name = cleaned_name[:-len(suffix)]
    
    return cleaned_name


def extract_metadata(filename: str) -> Optional[Dict[str, str]]:
    """
    Extract metadata from filename using regex pattern.
    
    Args:
        filename: Image filename to extract metadata from
        
    Returns:
        Dictionary of metadata if successful, None otherwise
    """
    match = re.search(FILENAME_PATTERN, filename)
    if match:
        return {
            'study_id': match.group(1),
            'biopsy_num': match.group(2),
            'step_size': match.group(3),
            'expo_time': match.group(4),
            'frame_number': match.group(5)
        }
    return None


def find_source_file(
    source_dir: str, 
    study_id: str, 
    biopsy_num: str, 
    image_name: str
) -> Optional[str]:
    """
    Find the source file path with various directory and extension checks.
    
    Args:
        source_dir: Base source directory
        study_id: Study ID
        biopsy_num: Biopsy number
        image_name: Original image name
        
    Returns:
        Source file path if found, None otherwise
    """
    base_name = clean_image_name(image_name)
    
    # Handle special cases in biopsy directory names
    actual_biopsy_dir = biopsy_num
    # Try different variations of biopsy directory names
    if not os.path.exists(os.path.join(source_dir, study_id, biopsy_num)):
        # Try removing any non-numeric characters
        clean_biopsy = re.sub(r'[^0-9]', '', biopsy_num)
        if os.path.exists(os.path.join(source_dir, study_id, f'Biopsy{clean_biopsy}')):
            actual_biopsy_dir = f'Biopsy{clean_biopsy}'
        elif 'a' in biopsy_num and os.path.exists(os.path.join(source_dir, study_id, biopsy_num.replace('a', ''))):
            actual_biopsy_dir = biopsy_num.replace('a', '')
    
    # Generate potential paths with different extensions
    potential_paths = [
        os.path.join(source_dir, study_id, actual_biopsy_dir, f"{base_name}{ext}") for ext in FILE_EXTENSIONS
    ]
    
    # Also check with the original image name (in case it already has an extension)
    potential_paths.append(os.path.join(source_dir, study_id, actual_biopsy_dir, image_name))
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_image_files_recursive(input_dir: str, supported_formats: list[str], target_suffix: Optional[str] = None) -> list[Path]:
    """Find image files recursively, optionally filtering by exact suffix."""
    input_path = Path(input_dir).resolve(); found_files = []
    if not input_path.is_dir(): logger.error(f"Input directory not found: {input_dir}"); return []
    search_desc = f"files ending with '{target_suffix}'" if target_suffix else f"images ({', '.join(supported_formats)})"
    logger.info(f"Recursively searching for {search_desc} in: {input_path}")
    initial_check_formats = supported_formats if target_suffix is None else ([target_suffix] if target_suffix else []) # Handle empty suffix
    if not initial_check_formats: initial_check_formats = supported_formats # Fallback if suffix logic fails
    for root, _, files in os.walk(input_path):
        for file in files:
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in initial_check_formats):
                if target_suffix is not None and not file.endswith(target_suffix): continue
                try:
                    file_path = Path(os.path.join(root, file))
                    if file_path.is_file(): found_files.append(file_path)
                except OSError as e: logger.warning(f"Could not access file {file_path}: {e}")
    if not found_files: logger.warning(f"No {search_desc} found in {input_dir}.")
    else: logger.debug(f"Found {len(found_files)} {search_desc}.")
    return sorted(found_files)


def process_fallback(
    source_path: str,
    base_name: str,
    label: str,
    images_dir: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a file using fallback method when regular metadata extraction fails.
    
    Args:
        source_path: Path to source file
        base_name: Cleaned base name of the file
        label: Classification label
        images_dir: Directory to save renamed images
        
    Returns:
        Tuple of (annotation_data, metadata_data) or (None, None) if failed
    """
    parts = base_name.split('StepSize')
    if len(parts) != 2:
        return None, None
        
    study_biopsy = parts[0]
    rest = parts[1]
    
    # Extract study_id and biopsy_num from the first part
    study_match = re.search(r'Study(LSM\d+)Biopsy(\d+\w?)', study_biopsy)
    if not study_match:
        return None, None
        
    study_id_match = study_match.group(1)
    biopsy_num_match = study_match.group(2)
    
    # Try to extract step_size, expo_time and frame_number
    rest_match = re.search(r'(\d+\.\d+)ExpoTime(\d+\.\d+)ms\((\d+)\)', rest)
    if not rest_match:
        return None, None
        
    step_size = rest_match.group(1)
    expo_time = rest_match.group(2)
    frame_number = rest_match.group(3)
    
    # Clean up any special characters in the biopsy number
    clean_biopsy = re.sub(r'[^0-9]', '', biopsy_num_match)
    new_filename = f"{study_id_match}_Biopsy{clean_biopsy}_Frame{frame_number}.tif"
    
    # Define the destination path
    dest_path = os.path.join(images_dir, new_filename)
    
    # Copy the file
    logger.info(f"Copying {source_path} to {dest_path}")
    try:
        shutil.copy2(source_path, dest_path)
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return None, None
    
    # Create data dictionaries
    annotation_data = {
        'image_path': os.path.join('images', new_filename),
        'label': label
    }
    
    metadata_data = {
        'image_path': os.path.join('images', new_filename),
        'study_id': study_id_match,
        'biopsy_num': clean_biopsy,
        'step_size': float(step_size),
        'expo_time': float(expo_time),
        'frame_number': int(frame_number)
    }
    
    return annotation_data, metadata_data


def create_output_subdir(output_base: Path, input_path: Path, input_base: Path) -> Tuple[Path, str]:
    """Create output subdirectory preserving input directory structure."""
    # Get relative path from input base to the file's parent directory
    rel_path = input_path.parent.relative_to(input_base) if input_base in input_path.parents else Path("")
    
    # Create output subdirectory
    output_subdir = output_base / rel_path
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Return output subdirectory and file stem
    return output_subdir, input_path.stem


def get_output_paths(output_subdir: Path, filename_stem: str, save_opts: Dict[str, bool] = None) -> Dict[str, Path]:
    """Generate all output paths with subdirectory organization."""
    # Default save options if none provided
    if save_opts is None:
        save_opts = {'normalized': True, 'entropy': True, 'depth': True, 'rgb': True, 'stacked': True, 'debug': True}
    
    # Create subdirectories for different output types
    channels_dir = output_subdir / "channels"
    normalized_dir = output_subdir / "normalized"
    rgb_dir = output_subdir / "rgb"
    debug_dir = output_subdir / "debug"
    
    # Only create directories that will be used
    directories_to_create = []
    
    # Check which directories need to be created based on save options
    if save_opts.get('normalized', False) or save_opts.get('entropy', False) or save_opts.get('depth', False):
        directories_to_create.append(channels_dir)
    
    if save_opts.get('rgb', False):
        directories_to_create.append(rgb_dir)
    
    if save_opts.get('stacked', False):
        directories_to_create.append(normalized_dir)
    
    if save_opts.get('debug', False):
        directories_to_create.append(debug_dir)
    
    # Create only the needed directories
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        # Channel data in channels directory
        'entropy': channels_dir / f"{filename_stem}{STATS_CHANNEL_SUFFIXES['entropy']}",
        'depth': channels_dir / f"{filename_stem}{STATS_CHANNEL_SUFFIXES['depth']}",
        
        # Regular normalized data in channels directory with other channel data
        'normalized': channels_dir / f"{filename_stem}{STATS_CHANNEL_SUFFIXES['normalized']}",
        
        # RGB encoded visualization in rgb directory
        'rgb': rgb_dir / f"{filename_stem}.png",
        
        # Stacked normalized in normalized directory
        'stacked': normalized_dir / f"{filename_stem}.png",
        
        # Debug visualizations in debug directory
        'debug': debug_dir / f"{filename_stem}.jpg"
    }


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalizes array to 0-255 range and converts to uint8."""
    if data is None or data.size == 0: return np.zeros((1, 1), dtype=np.uint8)
    min_val, max_val = np.min(data), np.max(data)
    if max_val <= min_val: return np.zeros_like(data, dtype=np.uint8)
    epsilon = np.finfo(float).eps
    normalized = np.clip((data - min_val) / (max_val - min_val + epsilon), 0, 1)
    return (normalized * 255).astype(np.uint8)


def normalize_exposure(img: np.ndarray, exposure_time: float, reference_exposure_time: float) -> np.ndarray:
    """Correct image intensity based on exposure time. Returns float32."""
    if exposure_time <= 0: logger.warning(f"Exposure time ({exposure_time}) invalid."); return img.astype(np.float32)
    img_float = img.astype(np.float32)
    scaling_factor = reference_exposure_time / exposure_time
    return img_float * scaling_factor


def apply_surface_mask(image: np.ndarray, surface_mask: np.ndarray) -> np.ndarray:
    """Apply surface mask, zeroing out pixels above the surface."""
    if image is None or surface_mask is None: logger.warning("apply_surface_mask: Image or mask None."); return image
    if image.shape[:2] != surface_mask.shape: raise ValueError(f"Shape mismatch: Image {image.shape[:2]} vs Mask {surface_mask.shape}")
    masked_image = image.copy()
    if masked_image.ndim == 2: masked_image[~surface_mask] = 0
    elif masked_image.ndim == 3: masked_image[~surface_mask, :] = 0
    else: logger.warning(f"apply_surface_mask: Unsupported dimensions ({image.ndim}).")
    return masked_image


# =============================================================================
# == CORE PROCESSING FUNCTIONS
# =============================================================================

def calculate_entropy(img: np.ndarray,
                      disk_radius: int = DEFAULT_ENTROPY_DISK_RADIUS,
                      sigma: float = DEFAULT_ENTROPY_SIGMA,
                      median_size: int = DEFAULT_ENTROPY_MEDIAN_SIZE,
                      lower_percentile: float = DEFAULT_ENTROPY_LOWER_PERCENTILE,
                      upper_percentile: float = DEFAULT_ENTROPY_UPPER_PERCENTILE,
                      use_ahe: bool = DEFAULT_USE_AHE,
                      ahe_clip_limit: float = DEFAULT_AHE_CLIP_LIMIT,
                      gamma: float = 1.0
                      ) -> np.ndarray:
    """Calculate local entropy of an image with preprocessing steps."""
    if img is None or img.size == 0: logger.warning("Entropy: Input None or empty."); return np.array([], dtype=np.float32)
    try:
        img_gray = rgb2gray(img) if img.ndim == 3 and img.shape[2] > 1 else np.squeeze(img, axis=2) if img.ndim == 3 else img
        img_float = img_as_float32(img_gray)
    except Exception as e: logger.error(f"Entropy: Conversion error: {e}"); return np.zeros(img.shape[:2] if img.ndim>=2 else (0,0), dtype=np.float32)
    epsilon = np.finfo(float).eps
    if np.abs(np.max(img_float) - np.min(img_float)) < epsilon: logger.debug("Entropy: Constant input."); return np.zeros_like(img_float)
    processed_img = img_float
    if sigma > 0:
        try: processed_img = gaussian(processed_img, sigma=sigma, preserve_range=True).astype(np.float32); logger.debug(f"Entropy: Gaussian (s={sigma})")
        except Exception as e: logger.warning(f"Entropy: Gaussian failed: {e}")
    if use_ahe:
        try:
            img_min, img_max = np.min(processed_img), np.max(processed_img)
            if img_max > img_min: norm_ahe = (processed_img-img_min)/(img_max-img_min+epsilon); processed_img = equalize_adapthist(norm_ahe, clip_limit=ahe_clip_limit).astype(np.float32); logger.debug(f"Entropy: AHE (clip={ahe_clip_limit})")
            else: logger.debug("Entropy: Skipping AHE (zero range).")
        except Exception as e: logger.warning(f"Entropy: AHE failed: {e}")
    if median_size > 1:
        if median_size % 2 == 0: median_size += 1
        try: processed_img = median_filter(processed_img, size=(median_size, 1), mode='reflect'); logger.debug(f"Entropy: Vertical median (sz={median_size})")
        except Exception as e: logger.warning(f"Entropy: Median failed: {e}")
    if np.abs(np.max(processed_img) - np.min(processed_img)) < epsilon: logger.debug("Entropy: Constant after processing."); return np.zeros_like(img_float)
    # processed_img = np.clip(processed_img, 0.0, 1.0)
    #scale to 0-255
    processed_img = (processed_img - np.min(processed_img)) / (np.max(processed_img) - np.min(processed_img))
    try: img_uint8 = img_as_ubyte(processed_img)
    except Exception as e: logger.error(f"Entropy: uint8 conversion failed: {e}"); return np.zeros_like(img_float)
    # Adjust disk radius dynamically if needed
    min_dim_shape = min(img_uint8.shape) if img_uint8.ndim == 2 else 0
    safe_radius = disk_radius
    if min_dim_shape > 0 and disk_radius >= min_dim_shape // 2:
        safe_radius = max(1, (min_dim_shape // 2) - 1)
        logger.warning(f"Entropy: Disk radius {disk_radius} too large for image shape {img_uint8.shape}. Adjusting to {safe_radius}.")
    if safe_radius < 1: safe_radius = 1

    try: 
        entropy_img = skimage_entropy(img_uint8, disk(safe_radius))
        # Apply gamma correction to the entropy image
        if gamma != 1.0:
            # Normalize to 0-1 range before applying gamma
            min_val, max_val = np.min(entropy_img), np.max(entropy_img)
            if max_val > min_val:
                normalized = (entropy_img - min_val) / (max_val - min_val)
                # Apply gamma correction
                gamma_corrected = np.power(normalized, gamma)
                # Scale back to original range
                entropy_img = min_val + (max_val - min_val) * gamma_corrected
                logger.debug(f"Entropy: Applied gamma correction (γ={gamma})")
        return entropy_img.astype(np.float32)
    except Exception as e: logger.error(f"Entropy: Calculation failed: {e}"); return np.zeros_like(img_float)


def _create_entropy_mask(entropy_map: np.ndarray, threshold_percentile=85, closing_radius=15, dilation_radius=5) -> np.ndarray:
    """Create a binary mask of high entropy regions with morphological cleaning."""
    if entropy_map is None or entropy_map.size == 0: return np.zeros((1, 1), dtype=bool)
    non_zero = entropy_map[entropy_map > 0]; threshold = np.percentile(non_zero, threshold_percentile) if non_zero.size > 0 else np.max(entropy_map)
    mask = entropy_map > threshold
    if closing_radius > 0: mask = binary_closing(mask, structure=disk(closing_radius))
    if dilation_radius > 0: mask = binary_dilation(mask, structure=disk(dilation_radius))
    mask = binary_fill_holes(mask); logger.debug(f"Created initial entropy mask (p={threshold_percentile})"); return mask


def _select_main_tissue_component(binary_mask: np.ndarray) -> Optional[np.ndarray]:
    """Identifies the main tissue component using connected components and heuristics."""
    if binary_mask is None or not np.any(binary_mask): return None
    rows, cols = binary_mask.shape; labeled_mask, num_comps = label(binary_mask)
    if num_comps == 0: return None
    props = regionprops(labeled_mask); candidates, scores = [], []; min_area, min_h_frac = 1000, 0.15
    for p in props:
        if p.area < min_area: continue
        min_r, min_c, max_r, max_c = p.bbox; h, w = max_r - min_r, max_c - min_c
        if h <= 0 or w <= 0 or h < rows * min_h_frac: continue
        a_s, b_s = p.area/(rows*cols), max_r/rows; asp = w/max(1, h); asp_s = 1.0 if (0.2<=asp<=5.0) else 0.5
        pos_s = 1.0 - abs((min_c+max_c)/2 - cols/2)/(cols/2) if cols > 0 else 1.0 # Handle cols=0 case
        score = (0.5*a_s + 0.3*b_s + 0.1*asp_s + 0.1*pos_s)
        candidates.append(p.label); scores.append(score)
    if not candidates:
        largest = max(props, key=lambda p: p.area, default=None)
        if largest and largest.area > min_area/2: logger.debug(f"Select Main: Fallback largest (L{largest.label})"); return labeled_mask == largest.label
        logger.warning("Select Main: No suitable component found."); return None
    selected_lbl = candidates[np.argmax(scores)]; logger.debug(f"Select Main: Selected L{selected_lbl} (score {np.max(scores):.3f})")
    return labeled_mask == selected_lbl


def _extract_surface_contour(main_tissue_mask: np.ndarray, smoothing_window_size: int = DEFAULT_CONTOUR_SMOOTHING_WINDOW) -> Optional[np.ndarray]:
    """Extracts surface contour directly from the main tissue mask with edge handling."""
    if main_tissue_mask is None or not np.any(main_tissue_mask): logger.warning("Contour: Input mask None or empty."); return None
    rows, cols = main_tissue_mask.shape
    if cols == 0: logger.warning("Contour: Input mask has zero columns."); return None
    first_true = np.argmax(main_tissue_mask, axis=0); has_true = np.any(main_tissue_mask, axis=0)
    valid_cols = np.where(has_true)[0]
    if len(valid_cols) == 0: logger.warning("Contour: No surface points found."); return None
    x_pts, y_pts = valid_cols, first_true[valid_cols].astype(float)
    min_r, min_a = 0.20, 5
    if len(x_pts) < min_a or len(x_pts) < cols * min_r: logger.warning(f"Contour: Too few valid points ({len(x_pts)})."); return None
    all_x = np.arange(cols); all_y = np.zeros(cols, dtype=float)
    if len(x_pts) == cols: all_y = y_pts
    else:
        if len(np.unique(x_pts)) < 2: logger.warning("Contour: < 2 unique x-coords for interp."); return None
        try:
            all_y = np.interp(all_x, x_pts, y_pts); first_x, last_x = x_pts[0], x_pts[-1]
            if first_x > 0:
                n_s = min(5, len(x_pts)); dx, dy = (x_pts[n_s-1]-x_pts[0]), (y_pts[n_s-1]-y_pts[0]) if n_s>=2 else (0,0); slope = dy/dx if dx!=0 else 0
                all_y[0:first_x] = y_pts[0] + slope * (all_x[0:first_x] - first_x)
            if last_x < cols-1:
                n_s = min(5, len(x_pts)); dx, dy = (x_pts[-1]-x_pts[-n_s]), (y_pts[-1]-y_pts[-n_s]) if n_s>=2 else (0,0); slope = dy/dx if dx!=0 else 0
                all_y[last_x+1:cols] = y_pts[-1] + slope * (all_x[last_x+1:cols] - last_x)
        except Exception as e: logger.error(f"Contour: Interp/extrap error: {e}"); return None
    if smoothing_window_size > 1 and cols >= smoothing_window_size:
        if smoothing_window_size%2==0: smoothing_window_size+=1
        try: all_y = uniform_filter1d(all_y, size=smoothing_window_size, mode='nearest')
        except Exception as e: logger.warning(f"Contour: Smoothing failed: {e}.")
    all_y_final = np.clip(np.round(all_y), 0, rows-1).astype(int); contour = np.column_stack((all_x, all_y_final))
    if contour.shape != (cols, 2): logger.error(f"Contour: Final shape incorrect ({contour.shape})."); return None
    logger.debug("Contour Extraction: Success."); return contour


def detect_tissue_surface_entropy(entropy_map: np.ndarray) -> Dict[str, Any]:
    """Detect the tissue surface using entropy (artifact removal skipped)."""
    if entropy_map is None: return {'surface_contour': None, 'binary_mask': None, 'main_tissue_mask': None}
    logger.info("Starting surface detection (artifact removal skipped).")
    binary_mask = _create_entropy_mask(entropy_map)
    shape = entropy_map.shape if entropy_map is not None else (0,0)
    if not np.any(binary_mask): return {'surface_contour': None, 'binary_mask': np.zeros(shape, dtype=bool), 'main_tissue_mask': np.zeros(shape, dtype=bool)}
    main_mask = _select_main_tissue_component(binary_mask)
    if main_mask is None or not np.any(main_mask): logger.warning("Surface Detect: Failed to select main component."); return {'surface_contour': None, 'binary_mask': binary_mask, 'main_tissue_mask': np.zeros_like(binary_mask)}
    contour = _extract_surface_contour(main_mask)
    if contour is None: logger.warning("Surface Detect: Failed to extract contour.")
    return {'surface_contour': contour, 'binary_mask': binary_mask, 'main_tissue_mask': main_mask}


def create_surface_mask(surface_contour: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Create a binary mask where True = tissue area (at and below surface contour)."""
    if surface_contour is None: return np.ones(shape, dtype=bool)
    rows, cols = shape
    if surface_contour.shape[0] != cols or surface_contour.shape[1] != 2:
        logger.error(f"Create Surface Mask: Invalid contour shape {surface_contour.shape}, expected ({cols}, 2).")
        return np.ones(shape, dtype=bool)
    y_coords = np.clip(surface_contour[:, 1].astype(int), 0, rows-1); rr, cc = np.ogrid[:rows, :cols]
    try: return rr >= y_coords[cc]
    except IndexError:
         logger.error("Create Surface Mask: Indexing error. Falling back to loop.")
         mask = np.zeros(shape, dtype=bool); x_coords = surface_contour[:,0].astype(int)
         for x, y in zip(x_coords, y_coords):
             if 0 <= x < cols: mask[y:, x] = True
         return mask


def calculate_depth_map(surface_contour: Optional[np.ndarray], shape: Tuple[int, int], max_depth: int = DEFAULT_MAX_DEPTH_MAP) -> np.ndarray:
    """Calculate an INVERTED L2 distance depth map from the surface contour."""
    rows, cols = shape; 
    if surface_contour is None: return np.zeros(shape, dtype=np.float32)
    if surface_contour.shape[0] != cols or surface_contour.shape[1] != 2: return np.zeros(shape, dtype=np.float32)
    tissue_mask = create_surface_mask(surface_contour, shape); 
    if not np.any(tissue_mask): return np.zeros(shape, dtype=np.float32)
    depth_raw = distance_transform_edt(tissue_mask); depth_clip = np.clip(depth_raw, 0, max_depth)
    inv_depth = np.zeros_like(depth_clip, dtype=np.float32); tissue_idx = np.where(tissue_mask)
    inv_depth[tissue_idx] = max_depth - depth_clip[tissue_idx]
    non_zero = inv_depth[tissue_idx]; logger.debug(f"Depth stats - Min:{np.min(non_zero):.2f}, Max:{np.max(inv_depth):.2f}, Mean:{np.mean(non_zero):.2f}") if non_zero.size > 0 else logger.warning("Depth map empty after inversion.")
    return inv_depth


# =============================================================================
# == VISUALIZATION FUNCTION
# =============================================================================

def create_debug_visualization(
    original_img: np.ndarray, normalized_img: np.ndarray, normalized_masked_img: np.ndarray,
    entropy_map: np.ndarray, depth_map: np.ndarray, rgb_encoded_image: np.ndarray,
    surface_contour: Optional[np.ndarray], main_tissue_mask: Optional[np.ndarray],
    surface_mask: Optional[np.ndarray], binary_mask: Optional[np.ndarray],
    save_path: Path):
    """Creates a comprehensive visualization showing key steps (artifact removal plot removed)."""
    if not MATPLOTLIB_AVAILABLE: return
    epsilon = np.finfo(float).eps; fig, axes = plt.subplots(2, 5, figsize=(25, 10)); fig.suptitle(f'Integrated Processing: {save_path.stem}', fontsize=16); axes = axes.flatten()
    def normalize_display(img_data):
        if img_data is None: return None
        min_v, max_v = np.min(img_data), np.max(img_data)
        return (img_data - min_v) / (max_v - min_v + epsilon) if max_v > min_v else np.zeros_like(img_data)
    titles = ['1. Original', '2. Normalized', '3. Surface Detect', '4. Surface Mask Overlay', '5. After Masking', '6. Entropy', '7. Initial Mask', '8. Main Tissue Mask', '9. Depth Map', '10. RGB Encoded']
    data = [normalize_display(original_img), normalize_display(normalized_img), normalize_display(normalized_img), normalize_display(normalized_img), normalize_display(normalized_masked_img), normalize_display(entropy_map), binary_mask, main_tissue_mask, normalize_display(depth_map), rgb_encoded_image]
    for i, (ax, title, img) in enumerate(zip(axes, titles, data)):
        ax.set_title(title); ax.axis('off')
        if img is not None:
            cmap = 'gray' if img.ndim == 2 and i != 9 else None; vmin, vmax = (0.0, 1.0) if img.ndim == 2 and i != 9 else (None, None)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            if title == '3. Surface Detect' and surface_contour is not None: ax.plot(surface_contour[:, 0], surface_contour[:, 1], 'r-', lw=1.5)
            elif title == '4. Surface Mask Overlay' and surface_mask is not None: overlay = np.zeros((*surface_mask.shape, 3), dtype=np.float32); overlay[~surface_mask, 0]=1.0; overlay[surface_mask, 1]=1.0; ax.imshow(overlay, alpha=0.3)
        elif img is None: ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    try: plt.savefig(save_path.with_suffix('.jpg'), dpi=300, bbox_inches='tight', format='jpg'); logger.debug(f" Saved debug: {save_path.with_suffix('.jpg').name}")
    except Exception as e: logger.error(f" Failed to save debug plot {save_path}: {e}")
    finally:
         if MATPLOTLIB_AVAILABLE: plt.close(fig)


# =============================================================================
# == DATASET STATISTICS CALCULATION FUNCTIONS
# =============================================================================

def calculate_stats_for_file(filepath: Path) -> Optional[Tuple[float, float, int]]:
    """Worker function: Reads an image file, calculates sum, sum of squares, and count of non-zero pixels."""
    try:
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None: logger.warning(f"Stats Worker: Could not read file: {filepath}"); return None
        
        # Convert image to float64 for accurate statistics
        # For PNG files, we need to normalize back from uint8 to original range
        img_float = img.astype(np.float64)
        
        # Skip zero pixels (typically background)
        non_zero = img_float[img_float != 0]
        if non_zero.size == 0: return 0.0, 0.0, 0
        
        return np.sum(non_zero), np.sum(non_zero**2), non_zero.size
    except Exception as e: logger.error(f"Stats Worker: Error processing file {filepath.name}: {e}", exc_info=False); return None


def calculate_dataset_channel_stats(output_dir: str, channel_suffix: str, max_workers: Optional[int] = None) -> Optional[Tuple[float, float]]:
    """Calculates the mean and standard deviation for a specific channel across all processed files."""
    output_path = Path(output_dir)
    # Look in the channels subdirectory for the channel files
    channels_dir = output_path / "channels"
    if not channels_dir.is_dir():
        logger.warning(f"Channels directory not found: {channels_dir}. Falling back to output directory.")
        channels_dir = output_path
    
    logger.info(f"Calculating dataset statistics for {channel_suffix} files in {channels_dir}...")
    # Skip initial brackets when searching for files with the suffix
    clean_suffix = channel_suffix.replace("[", "").replace("]", "")
    image_files = find_image_files_recursive(str(channels_dir), [], target_suffix=clean_suffix)
    if not image_files:
        logger.warning(f"No {channel_suffix} files found in {channels_dir}.")
        return None
    
    logger.info(f"Found {len(image_files)} {channel_suffix} files for statistics calculation.")
    if max_workers is None: max_workers = max(1, mp.cpu_count() - 1)
    
    try:
        total_sum, total_sum_sq, total_count = 0, 0, 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(calculate_stats_for_file, img_path) for img_path in image_files]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        pixel_sum, pixel_sum_sq, pixel_count = result
                        total_sum += pixel_sum
                        total_sum_sq += pixel_sum_sq
                        total_count += pixel_count
                except Exception as e:
                    logger.error(f"Error calculating stats: {e}")
        
        if total_count == 0:
            logger.warning(f"No valid pixels found in {channel_suffix} files.")
            return None
        
        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean * mean)
        std_dev = np.sqrt(max(0, variance))  # Avoid negative variance due to numerical issues
        
        # Save statistics to a file
        stats_file = output_path / f"dataset_stats_{channel_suffix.replace('.', '')}.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Channel: {channel_suffix}\n")
            f.write(f"Mean: {mean}\n")
            f.write(f"Standard Deviation: {std_dev}\n")
            f.write(f"Total Pixels: {total_count}\n")
            f.write(f"Files Analyzed: {len(image_files)}\n")
        
        logger.info(f"Statistics for {channel_suffix}: Mean={mean:.6f}, StdDev={std_dev:.6f}")
        logger.info(f"Saved statistics to {stats_file}")
        
        return mean, std_dev
    except Exception as e:
        logger.error(f"Failed to calculate statistics for {channel_suffix}: {e}")
        return None


# =============================================================================
# == PREPROCESSING PIPELINE INTEGRATION FUNCTIONS
# =============================================================================

def rename_and_collect_metadata(
    source_dir: str,
    annotations_file: str,
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Processes the original dataset, renames files, and extracts metadata.
    
    Args:
        source_dir: Base directory with original images
        annotations_file: Path to original annotations CSV
        output_dir: Base output directory
        
    Returns:
        Tuple of (annotations_df, metadata_df, processing_tasks)
    """
    logger.info(f"Starting dataset renaming and metadata extraction")
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the original annotations file
    try:
        df = pd.read_csv(annotations_file)
        logger.info(f"Loaded annotations with {len(df)} images")
    except Exception as e:
        logger.error(f"Error loading annotations file: {e}")
        raise
    
    # Initialize lists to store new data
    new_annotations = []
    new_metadata = []
    processing_tasks = []
    
    # Process each row in the original annotations file
    for index, row in df.iterrows():
        study_id = row['study_id']
        biopsy_num = row['biopsy_num']
        image_name = row['image_name']
        label = row['label']
        
        # Find the source file
        source_path = find_source_file(source_dir, study_id, biopsy_num, image_name)
        
        if not source_path:
            logger.warning(f"Could not find source file for {image_name} in {study_id}/{biopsy_num}")
            continue
        
        # Get base name for metadata extraction
        base_name = clean_image_name(image_name)
        
        # Get the file extension from the source path
        file_extension = os.path.splitext(source_path)[1]
        
        # Extract metadata using the extraction function
        metadata = extract_metadata(base_name)
        
        if metadata:
            # Extract components
            study_id_match = metadata['study_id']
            biopsy_num_match = metadata['biopsy_num']
            step_size = metadata['step_size']
            expo_time = metadata['expo_time']
            frame_number = metadata['frame_number']
            # TODO: Check if 1a for example is wanted
            # Clean up any special characters in the biopsy number
            clean_biopsy = re.sub(r'[^0-9]', '', biopsy_num_match)
            
            # Create standardized filename stem and full filename
            new_filename_stem = f"{study_id_match}_{biopsy_num_match}_{round(float(expo_time) * 1000)}_{frame_number}"
            new_filename = f"{new_filename_stem}{file_extension}"
            
            # Define the destination path
            dest_path = images_dir / new_filename
            
            # Copy the file
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {os.path.basename(source_path)} to {new_filename}")
                
                # Add to the new annotations list
                new_annotations.append({
                    'image_path': f"{new_filename}",
                    'label': label
                })
                
                # Add to the metadata list
                new_metadata.append({
                    'image_path': f"{new_filename}",
                    'study_id': study_id_match,
                    'biopsy_num': biopsy_num_match,
                    'step_size': float(step_size),
                    'expo_time': float(expo_time),
                    'frame_number': int(frame_number)
                })
                
                # Add to processing tasks for later parallel processing
                processing_tasks.append({
                    'source_path': Path(source_path),
                    'output_subdir': output_path,
                    'filename_stem': new_filename_stem,
                    'label': label
                })
                
            except Exception as e:
                logger.error(f"Error copying file {source_path} to {dest_path}: {e}")
        else:
            logger.warning(f"Could not extract metadata from {base_name}, trying fallback method")
            # Try fallback approach
            annotation_data, metadata_data = process_fallback(source_path, base_name, label, str(images_dir))
            if annotation_data and metadata_data:
                new_annotations.append(annotation_data)
                new_metadata.append(metadata_data)
                
                # Extract filename from the path for processing tasks
                rel_path = annotation_data['image_path'].split('/')[-1]
                filename_stem = os.path.splitext(rel_path)[0]
                
                processing_tasks.append({
                    'source_path': Path(source_path),
                    'output_subdir': output_path,
                    'filename_stem': filename_stem,
                    'label': label
                })
            else:
                logger.error(f"Failed to process {image_name} with fallback method")
    
    # Create DataFrames from the collected data
    annotations_df = pd.DataFrame(new_annotations)
    metadata_df = pd.DataFrame(new_metadata)
    
    # Save the new annotations files
    try:
        annotations_df.to_csv(output_path / 'annotations.csv', index=False)
        metadata_df.to_csv(output_path / 'metadata.csv', index=False)
        logger.info(f"Saved annotations to {output_path / 'annotations.csv'}")
        logger.info(f"Saved metadata to {output_path / 'metadata.csv'}")
    except Exception as e:
        logger.error(f"Error saving CSV files: {e}")
        raise
    
    logger.info(f"Renaming and metadata extraction complete!")
    logger.info(f"Original dataset had {len(df)} images")
    logger.info(f"Renamed dataset has {len(annotations_df)} images")
    
    return annotations_df, metadata_df, processing_tasks


# =============================================================================
# == MAIN PROCESSING PIPELINE FUNCTIONS
# =============================================================================

def process_image(task_obj: Dict[str, Any], save_opts: Dict[str, bool] = None, metadata_csv_path: str = None) -> bool:
    """
    Processes a single image using the integrated pipeline.
    
    Args:
        task_obj: Dictionary containing task information:
                  - source_path: Path to source image
                  - output_subdir: Output subdirectory
                  - filename_stem: Filename stem for outputs
        save_opts: Dictionary of save options
        
    Returns:
        True if processing was successful, False otherwise
    """
    if save_opts is None:
        save_opts = {
            'normalized': True, 
            'entropy': True, 
            'depth': True, 
            'rgb': True, 
            'stacked': True, 
            'debug': True
        }
    
    img_path = task_obj['source_path']
    output_subdir = task_obj['output_subdir']
    filename_stem = task_obj['filename_stem']
    
    start_time = time.time(); logger.info(f"Processing: {img_path.name}")
    try:
        # Get output paths and create only needed directories
        paths = get_output_paths(output_subdir, filename_stem, save_opts)
        
        # Read source image
        img_orig = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_orig is None: raise ValueError(f"Failed read: {img_path}")
        
        # Extract image properties and convert to grayscale if needed
        dtype_orig = img_orig.dtype
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) if img_orig.ndim==3 else img_orig
        shape_orig = img_gray.shape
        
        # Extract exposure time and normalize
        expo = extract_exposure_time(img_path.name, metadata_csv_path) or DEFAULT_EXPOSURE_TIME
        img_norm = normalize_exposure(img_gray, expo, DEFAULT_EXPOSURE_TIME)
        
        # Calculate entropy and detect tissue surface
        entropy = calculate_entropy(img_norm, gamma=DEFAULT_ENTROPY_GAMMA)
        surf_info = detect_tissue_surface_entropy(entropy)
        contour, main_mask, bin_mask = surf_info.get('surface_contour'), surf_info.get('main_tissue_mask'), surf_info.get('binary_mask')
        
        # Create surface mask and apply to images
        surf_mask = create_surface_mask(contour, shape_orig) if contour is not None else np.ones(shape_orig, dtype=bool)
        img_norm_masked = apply_surface_mask(img_norm, surf_mask)
        entropy_masked = apply_surface_mask(entropy, surf_mask)
        
        # Calculate depth map
        depth = calculate_depth_map(contour, shape_orig) if contour is not None else None
        
        # Initialize RGB and stacked images as None
        rgb = None
        stacked_img = None
        
        # Create RGB encoded image if needed
        if save_opts.get('rgb', False) or save_opts.get('debug', False):
            if depth is not None:
                n8 = normalize_to_uint8(img_norm_masked)
                e8 = normalize_to_uint8(entropy_masked)
                d8 = normalize_to_uint8(depth)
                rgb = cv2.merge([d8, e8, n8])
        
        # Create stacked normalized image (all 3 channels same) if needed
        if save_opts.get('stacked', False):
            img_norm_uint8 = normalize_to_uint8(img_norm)
            stacked_img = cv2.merge([img_norm_uint8, img_norm_uint8, img_norm_uint8])
        
        # Save outputs based on save options
        ok = True
        try:
            # Save normalized intensity channel
            if save_opts.get('normalized', False):
                norm_uint8 = normalize_to_uint8(img_norm_masked)
                if not cv2.imwrite(str(paths['normalized']), norm_uint8):
                     logger.error(f" Save failed: {paths['normalized']}"); ok=False
            
            # Save entropy channel
            if save_opts.get('entropy', False):
                # Convert entropy to uint8 explicitly for PNG saving
                entropy_uint8 = normalize_to_uint8(entropy_masked)
                if not cv2.imwrite(str(paths['entropy']), entropy_uint8):
                     logger.error(f" Save failed: {paths['entropy']}"); ok=False
                else:
                     logger.debug(f" Saved entropy as 8-bit PNG: {paths['entropy'].name}")
            
            # Save depth channel
            if save_opts.get('depth', False) and depth is not None:
                # Depth maps need to be explicitly converted to uint8 for PNG
                depth_uint8 = normalize_to_uint8(depth)
                # Force OpenCV to save as 8-bit single-channel image
                if not cv2.imwrite(str(paths['depth']), depth_uint8):
                     logger.error(f" Save failed: {paths['depth']}"); ok=False
                else:
                     logger.debug(f" Saved depth map as 8-bit PNG: {paths['depth'].name}")
            
            # Save RGB encoded image
            if save_opts.get('rgb', False) and rgb is not None:
                 if not cv2.imwrite(str(paths['rgb']), rgb):
                      logger.warning(f" Non-critical save failed: {paths['rgb']}")
            
            # Save stacked normalized image
            if save_opts.get('stacked', False) and stacked_img is not None:
                if not cv2.imwrite(str(paths['stacked']), stacked_img):
                    logger.warning(f" Non-critical save failed: {paths['stacked']}")
        except Exception as e: 
            logger.error(f"Exception saving outputs for {filename_stem}: {e}")
            ok=False
        
        # Create debug visualization if enabled
        if save_opts.get('debug', False) and MATPLOTLIB_AVAILABLE:
            try:
                if rgb is None and depth is not None:
                    n8 = normalize_to_uint8(img_norm_masked)
                    e8 = normalize_to_uint8(entropy_masked)
                    d8 = normalize_to_uint8(depth)
                    rgb = cv2.merge([d8, e8, n8])
                
                if rgb is not None:
                    create_debug_visualization(
                        img_gray, img_norm, img_norm_masked, 
                        entropy_masked, depth, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                        contour, main_mask, surf_mask, bin_mask, paths['debug']
                    )
            except Exception as e:
                logger.error(f"Error creating debug plot for {filename_stem}: {e}", exc_info=True)
        
        logger.info(f" Completed {img_path.name} in {time.time() - start_time:.2f}s")
        return ok
    except Exception as e:
        logger.error(f" UNHANDLED ERROR processing {img_path.name}: {e}", exc_info=True)
        return False


def process_images_parallel(
    processing_tasks: List[Dict[str, Any]],
    save_opts: Dict[str, bool],
    max_workers: Optional[int] = None,
    metadata_csv_path: str = None
) -> Tuple[int, int]:
    """
    Process multiple images in parallel.
    
    Args:
        processing_tasks: List of task dictionaries for image processing
        save_opts: Dictionary of save options
        max_workers: Number of parallel workers (None=auto)
        
    Returns:
        Tuple of (success_count, error_count)
    """
    if not processing_tasks: return 0, 0
    if max_workers is None: max_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Starting parallel processing: {len(processing_tasks)} images, {max_workers} workers.")
    success_count, error_count = 0, 0
    total = len(processing_tasks)
    start = time.time()
    last_log = start

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for task in processing_tasks:
            try:
                future = executor.submit(process_image, task, save_opts, metadata_csv_path)
                future_map[future] = task
            except Exception as e:
                img_path = task.get('source_path', 'Unknown')
                logger.error(f"Error submitting job for {img_path}: {e}")
                error_count += 1

        processed = error_count
        for future in as_completed(future_map):
            task = future_map[future]
            img_path = task.get('source_path', 'Unknown')
            processed += 1
            
            try:
                if future.result():
                    success_count += 1
                else:
                    error_count += 1
                    logger.warning(f"Processing issue for {img_path}")
            except Exception as e:
                error_count += 1
                logger.error(f"Exception processing {img_path}: {e}", exc_info=False)
            
            now = time.time()
            if processed % 20 == 0 or (now - last_log) > 10 or processed == total:
                elapsed = now - start
                rate = (processed - error_count) / max(elapsed, 0.1)
                eta = (total - processed) / max(rate, 0.1) if rate > 0 else 0
                logger.info(f"Progress: {processed}/{total} ({processed/total:.1%}) | Rate: {rate:.2f} img/s | ETA: {eta:.0f}s")
                last_log = now

    total_t = time.time() - start
    logger.info(f"Parallel processing complete: {success_count} succeeded, {error_count} errors in {total_t:.2f}s")
    if total_t > 0 and total > 0:
        logger.info(f"Avg rate: {success_count / total_t:.2f} successful img/sec")
    
    return success_count, error_count


# =============================================================================
# == MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main entry point for the integrated preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Integrated UA-SLSM Preprocessing Pipeline.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Input/output options
    parser.add_argument('source_directory', help='Source directory with original dataset structure')
    parser.add_argument('annotations_file', help='Path to original annotations.csv file')
    parser.add_argument('output_directory', help='Base output directory for processed data')
    parser.add_argument('--metadata-csv', help='Path to metadata CSV file with exposure time information')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing options')
    proc_group.add_argument('-j', '--jobs', type=int, default=None, help='Parallel processes (default: auto)')
    proc_group.add_argument('--sequential', action='store_true', help='Sequential processing')
    proc_group.add_argument('--limit', type=int, default=None, help='Limit to first N images')
    proc_group.add_argument('--skip-rename', action='store_true', help='Skip renaming step (use existing renamed dataset)')
    proc_group.add_argument('--calculate-stats', action='store_true', help='Calculate mean/std dev after processing')
    
    # Output options
    out_group = parser.add_argument_group('Output options')
    out_group.add_argument('--only', nargs='+', choices=['normalized', 'entropy', 'depth', 'rgb', 'stacked', 'debug'], 
                          help='Only save specified outputs')
    out_group.add_argument('--no-normalized', action='store_true', help='Skip saving normalized')
    out_group.add_argument('--no-entropy', action='store_true', help='Skip saving entropy')
    out_group.add_argument('--no-depth', action='store_true', help='Skip saving depth')
    out_group.add_argument('--no-rgb', action='store_true', help='Skip saving RGB')
    out_group.add_argument('--no-stacked', action='store_true', help='Skip saving stacked normalized image')
    out_group.add_argument('--no-debug', action='store_true', help='Skip saving debug plot')
    
    # Logging options
    log_group = parser.add_argument_group('Logging options')
    log_group.add_argument('-v', '--verbose', action='store_true', help='Verbose logging (DEBUG)')
    log_group.add_argument('-q', '--quiet', action='store_true', help='Quiet logging (WARNING)')
    
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logger.setLevel(log_level)
    
    # Configure save options
    if args.only:
        save_opts = {k: k in args.only for k in ['normalized', 'entropy', 'depth', 'rgb', 'stacked', 'debug']}
    else:
        save_opts = {
            'normalized': not args.no_normalized,
            'entropy': not args.no_entropy,
            'depth': not args.no_depth,
            'rgb': not args.no_rgb,
            'stacked': not args.no_stacked,
            'debug': not args.no_debug
        }
    
    enabled_outputs = ', '.join([k for k, v in save_opts.items() if v]) or "None"
    logger.info(f"Enabled outputs: {enabled_outputs}")
    
    if save_opts['debug'] and not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not installed, debug plots disabled.")
    
    # Setup input/output paths
    source_dir = Path(args.source_directory).resolve()
    output_dir = Path(args.output_directory).resolve()
    annotations_file = Path(args.annotations_file).resolve()
    metadata_csv_path = Path(args.metadata_csv).resolve() if args.metadata_csv else None
    
    # Load metadata CSV if provided
    if metadata_csv_path and metadata_csv_path.is_file():
        logger.info(f"Loading metadata from {metadata_csv_path}")
        load_metadata_csv(str(metadata_csv_path))
    elif args.metadata_csv:
        logger.warning(f"Metadata CSV file not found: {metadata_csv_path}")
        metadata_csv_path = None
    
    if not source_dir.is_dir():
        logger.critical(f"Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    if not annotations_file.is_file():
        logger.critical(f"Annotations file does not exist: {annotations_file}")
        sys.exit(1)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Cannot create output directory {output_dir}: {e}")
        sys.exit(1)
    
    start_run = time.time()
    processing_tasks = []
    
    # Step 1: Rename images and extract metadata
    if not args.skip_rename:
        logger.info("Step 1: Renaming images and extracting metadata")
        try:
            _, _, processing_tasks = rename_and_collect_metadata(
                str(source_dir),
                str(annotations_file),
                str(output_dir)
            )
            logger.info(f"Generated {len(processing_tasks)} processing tasks")
        except Exception as e:
            logger.critical(f"Renaming failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Skipping renaming step (using existing renamed dataset)")
        # If skipping rename, we need to generate processing tasks from the existing renamed dataset
        try:
            images_dir = source_dir
            # images_dir = output_dir
            if not images_dir.is_dir():
                logger.critical(f"Images directory not found: {images_dir}. Cannot skip renaming step.")
                sys.exit(1)
            
            image_files = find_image_files_recursive(str(images_dir), SUPPORTED_FORMATS)
            logger.info(f"Found {len(image_files)} images in existing renamed dataset")
            
            for img_path in image_files:
                processing_tasks.append({
                    'source_path': img_path,
                    'output_subdir': output_dir,
                    'filename_stem': img_path.stem,
                    'label': None  # Label might not be known
                })
        except Exception as e:
            logger.critical(f"Failed to process existing renamed dataset: {e}", exc_info=True)
            sys.exit(1)
    
    # Apply limit if specified
    if args.limit and 0 < args.limit < len(processing_tasks):
        logger.info(f"Limiting to first {args.limit} images")
        processing_tasks = processing_tasks[:args.limit]
    
    if not processing_tasks:
        logger.info("No images to process")
        sys.exit(0)
    
    # Step 2: Process images
    logger.info("Step 2: Processing images")
    num_workers = 1 if args.sequential else args.jobs
    
    if args.sequential:
        logger.info("Processing sequentially...")
        success_count = 0
        error_count = 0
        
        for i, task in enumerate(processing_tasks):
            logger.info(f"--- Image {i+1}/{len(processing_tasks)} ---")
            if process_image(task, save_opts, str(metadata_csv_path) if metadata_csv_path else None):
                success_count += 1
            else:
                error_count += 1
    else:
        # Parallel processing
        success_count, error_count = process_images_parallel(
            processing_tasks,
            save_opts,
            num_workers,
            str(metadata_csv_path) if metadata_csv_path else None
        )
    
    # Step 3: Calculate dataset statistics if requested
    if args.calculate_stats:
        logger.info("=" * 50 + "\nStep 3: Dataset Statistics Calculation" + "\n" + "=" * 50)
        stats_results = {}
        stats_workers = 1 if args.sequential else args.jobs
        
        for name, suffix in STATS_CHANNEL_SUFFIXES.items():
            if save_opts.get(name, False):
                stats = calculate_dataset_channel_stats(
                    str(output_dir),
                    suffix,
                    max_workers=stats_workers
                )
                stats_results[name] = stats
            else:
                logger.info(f"Skipping stats for '{name}' (not saved)")
        
        # Save comprehensive statistics summary
        stats_summary_file = output_dir / "dataset_statistics_summary.txt"
        with open(stats_summary_file, 'w') as f:
            f.write("=== UA-SLSM Dataset Statistics Summary ===\n\n")
            f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source Directory: {source_dir}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"Images Processed: {success_count} (of {len(processing_tasks)} attempted)\n\n")
            
            # Format channel stats
            norm_mean = stats_results.get('normalized', (None, None))[0] if 'normalized' in stats_results else None
            entropy_mean = stats_results.get('entropy', (None, None))[0] if 'entropy' in stats_results else None
            depth_mean = stats_results.get('depth', (None, None))[0] if 'depth' in stats_results else None
            
            norm_sd = stats_results.get('normalized', (None, None))[1] if 'normalized' in stats_results else None
            entropy_sd = stats_results.get('entropy', (None, None))[1] if 'entropy' in stats_results else None
            depth_sd = stats_results.get('depth', (None, None))[1] if 'depth' in stats_results else None
            
            f.write("Channel Statistics:\n")
            # Format values with proper conditional handling
            norm_mean_str = f"{norm_mean:.6f}" if norm_mean is not None else "N/A"
            entropy_mean_str = f"{entropy_mean:.6f}" if entropy_mean is not None else "N/A"
            depth_mean_str = f"{depth_mean:.6f}" if depth_mean is not None else "N/A"
            
            norm_sd_str = f"{norm_sd:.6f}" if norm_sd is not None else "N/A"
            entropy_sd_str = f"{entropy_sd:.6f}" if entropy_sd is not None else "N/A"
            depth_sd_str = f"{depth_sd:.6f}" if depth_sd is not None else "N/A"
            
            f.write(f"mean: [{norm_mean_str}, {entropy_mean_str}, {depth_mean_str}]\n")
            f.write(f"sd: [{norm_sd_str}, {entropy_sd_str}, {depth_sd_str}]\n")
            
            # Also include detailed stats
            f.write("\nDetailed Channel Statistics:\n")
            f.write("-" * 50 + "\n")
            for name, stats in stats_results.items():
                if stats:
                    f.write(f"{name.capitalize()} Channel:\n")
                    f.write(f"  Mean: {stats[0]:.6f}\n")
                    f.write(f"  Standard Deviation: {stats[1]:.6f}\n")
                    f.write("-" * 50 + "\n")
                else:
                    f.write(f"{name.capitalize()} Channel: Not calculated or failed\n")
                    f.write("-" * 50 + "\n")
        
        logger.info(f"Saved comprehensive statistics summary to {stats_summary_file}")
        logger.info("--- Final Dataset Statistics ---")
        for name, stats in stats_results.items():
            logger.info(f"  {name.capitalize():<12}: Mean={stats[0]:.6f}, StdDev={stats[1]:.6f}" if stats else f"  {name.capitalize():<12}: Not calculated/Failed.")
        logger.info("--- End Dataset Statistics ---")
    
    # Summary
    total_t = time.time() - start_run
    logger.info("=" * 50 + "\nProcessing Run Summary" + "\n" + "=" * 50)
    logger.info(f"  Images Attempted: {len(processing_tasks)}")
    logger.info(f"  Successes: {success_count}")
    logger.info(f"  Failures/Errors: {error_count}")
    logger.info(f"  Total Time: {total_t:.2f}s")
    logger.info(f"  Avg. Time/Success: {total_t / success_count:.2f}s" if success_count > 0 else "  Avg. Time/Success: N/A")
    logger.info("=" * 50)
    
    sys.exit(0 if error_count == 0 else 1)


if __name__ == '__main__':
    main()