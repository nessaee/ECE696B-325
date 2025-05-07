"""
Utility functions for the UA-SLSM MLP training pipeline.
Contains helper functions used across multiple modules.
"""

import torch
import numpy as np
import random
import json
from pathlib import Path
import logging

def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get the appropriate device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: Device to use for computation
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_json(data, filepath):
    """
    Save data to a JSON file with proper formatting.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath (str or Path): Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """
    Load data from a JSON file with error handling.
    
    Args:
        filepath (str or Path): Path to the JSON file
    
    Returns:
        The loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file (str or Path, optional): Path to log file
        log_level: Logging level (default: INFO)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
