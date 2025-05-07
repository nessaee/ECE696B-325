"""
Image renaming and metadata extraction utility.

This script processes images from the original dataset structure, renames them
according to a consistent format, and extracts metadata for later use.
"""

import os
import re
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Regular expression patterns to extract metadata
# Main pattern to match most file names
PATTERN = r'Study(LSM\d+)Biopsy(\d+\w?)(?:a)?StepSize(\d+\.\d+)ExpoTime(\d+\.\d+)ms\((\d+)\)'
# List of suffixes to remove
SUFFIXES = ['_rgb_invD.png', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
# Supported file extensions
EXTENSIONS = ['.tif', '.tiff', '.tiff.tif', '.png', '_rgb_invD.png', '.jpg', '.jpeg']


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
    for suffix in SUFFIXES:
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
    match = re.search(PATTERN, filename)
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
        os.path.join(source_dir, study_id, actual_biopsy_dir, f"{base_name}{ext}") for ext in EXTENSIONS
    ]
    
    # Also check with the original image name (in case it already has an extension)
    potential_paths.append(os.path.join(source_dir, study_id, actual_biopsy_dir, image_name))
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    return None


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


def main():
    """Main function for image renaming and metadata extraction."""
    # Define paths
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/rgb')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/renamed-dataset')
    images_dir = os.path.join(output_dir, 'images')
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Read the original annotations file
    try:
        df = pd.read_csv(os.path.join(source_dir, 'annotations.csv'))
        logger.info(f"Loaded annotations with {len(df)} images")
    except Exception as e:
        logger.error(f"Error loading annotations file: {e}")
        return
    
    # Initialize lists to store new data
    new_annotations = []
    new_metadata = []
    
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
            
            # Clean up any special characters in the biopsy number
            clean_biopsy = re.sub(r'[^0-9]', '', biopsy_num_match)
            new_filename = f"{study_id_match}_{biopsy_num_match}_Frame{frame_number}{file_extension}"
            
            # Define the destination path
            dest_path = os.path.join(images_dir, new_filename)
            
            # Copy the file
            try:
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {os.path.basename(source_path)} to {new_filename}")
                
                # Add to the new annotations list
                new_annotations.append({
                    'image_path': f"images/{new_filename}",
                    'label': label
                })
                
                # Add to the metadata list
                new_metadata.append({
                    'image_path': f"images/{new_filename}",
                    'study_id': study_id_match,
                    'biopsy_num': biopsy_num_match,
                    'step_size': float(step_size),
                    'expo_time': float(expo_time),
                    'frame_number': int(frame_number)
                })
            except Exception as e:
                logger.error(f"Error copying file {source_path} to {dest_path}: {e}")
        else:
            logger.warning(f"Could not extract metadata from {base_name}, trying fallback method")
            # Try fallback approach
            annotation_data, metadata_data = process_fallback(source_path, base_name, label, images_dir)
            if annotation_data and metadata_data:
                new_annotations.append(annotation_data)
                new_metadata.append(metadata_data)
            else:
                logger.error(f"Failed to process {image_name} with fallback method")
    
    # Create DataFrames from the collected data (more efficient than concatenating in the loop)
    new_df = pd.DataFrame(new_annotations)
    metadata_df = pd.DataFrame(new_metadata)
    
    # Save the new annotations files
    try:
        new_df.to_csv(os.path.join(output_dir, 'annotations.csv'), index=False)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        logger.info(f"Saved annotations to {os.path.join(output_dir, 'annotations.csv')}")
        logger.info(f"Saved metadata to {os.path.join(output_dir, 'metadata.csv')}")
    except Exception as e:
        logger.error(f"Error saving CSV files: {e}")
    
    logger.info(f"\nRenaming and metadata extraction complete!")
    logger.info(f"Original dataset had {len(df)} images")
    logger.info(f"Renamed dataset has {len(new_df)} images")
    logger.info(f"Images are stored in: {images_dir}")


if __name__ == "__main__":
    main()
