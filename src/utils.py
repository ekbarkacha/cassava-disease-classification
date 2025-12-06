"""
Module: utils.py
================

Author: Emmanuel & Tiana
Created: December 2025

Description:
------------
Utility configuration module for the Cassava Leaf Disease Classification project.

This module centralizes global constants, directory configurations, and device
selection logic used throughout the codebase. Keeping these settings in a single
location ensures consistency, reduces duplication, and improves maintainability
across training, inference, and dataset components.
"""

import os
import torch

from src.dataset import create_dataloaders

# Device Selection
def _select_device():
    """
    Select the best available compute device in the following order:
    1) Apple MPS (Metal Performance Shaders)
    2) CUDA GPU
    3) CPU fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = _select_device()

# Global Constants
NUM_CLASSES = 5
BATCH_SIZE = 8

AVAILABLE_MODELS = {"densenet":"DenseNet121","efficientnet":"EfficientNet-B4","convnext":"ConvNeXt"}

# Base dataset directory
DATA_DIR = 'data/cassava-disease-classification'

# Subdirectories for training, extra data, and testing
TRAIN_DIR = os.path.join(DATA_DIR, "train/train")
Exra_DIR = os.path.join(DATA_DIR, "extraimages/extraimages")
TEST_DIR = os.path.join(DATA_DIR, "test/test/0")

# Model weights & submissions
WEIGHTS_DIR = "models"
SUBMISSION_DIR = "submissions"
REPORT_DIR = "reports"



# Directory Validation/Initialization
def ensure_directories():
    """
    Ensure directories required by training/inference exist.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

ensure_directories()

# Get All Dataloaders
ALL_DATALOADERS_DATASET = create_dataloaders(train_dir=TRAIN_DIR,
                                             test_dir=TEST_DIR, 
                                             extra_dir=Exra_DIR, 
                                             batch_size=BATCH_SIZE)