"""
Module: inference.py
====================

Author: Emmanuel & Tiana
Created: December 2025

Description:
------------
Inference pipeline for Cassava Leaf Disease Classification.

This module provides an object-oriented framework for running model inference
using single deep learning models or ensemble techniques. It is designed to
integrate cleanly into the project's modular architecture and supports scalable,
production-ready prediction workflows.

Features:
---------
- ModelLoader class for loading EfficientNet-B4, ConvNeXt-Tiny and DenseNet121
  with pretrained or fine-tuned weights.
- Predictor class for single-model inference with batched prediction support.
- EnsemblePredictor class using softmax-probability averaging for robust,
  multi-model predictions.
- Submission utilities for generating Kaggle-compatible CSV outputs.
- A unified `run_inference()` entry point that orchestrates prediction flows
  for both single and ensemble models.
"""

import os
import torch
import glob
import argparse
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from src.utils import (NUM_CLASSES, SUBMISSION_DIR, 
                       WEIGHTS_DIR,DEVICE, TEST_DIR, 
                       ALL_DATALOADERS_DATASET)


import warnings
warnings.filterwarnings("ignore")

full_dataset = ALL_DATALOADERS_DATASET["full_dataset"]
submit_test_loader = ALL_DATALOADERS_DATASET["submit_test_loader"]

# Model Loader Class
class ModelLoader:

    def load_efficientnet_b4(self):
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        path = os.path.join(WEIGHTS_DIR, "EfficientNet-B4.pth")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model.to(DEVICE).eval()

    def load_convnext_tiny(self):
        model = models.convnext_tiny(weights=models.convnext.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
        path = os.path.join(WEIGHTS_DIR, "ConvNeXt.pth")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model.to(DEVICE).eval()

    def load_densenet121(self):
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        path = os.path.join(WEIGHTS_DIR, "DenseNet121.pth")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model.to(DEVICE).eval()


# Predictor Classes
class Predictor:
    """Single-model batched inference."""

    def __init__(self, model):
        self.model = model

    def predict(self, dataloader):
        filenames, preds = [], []

        self.model.eval()
        with torch.no_grad():
            for imgs, names in dataloader:
                imgs = imgs.to(DEVICE)
                outputs = self.model(imgs)

                labels = torch.argmax(outputs, dim=1)
                preds.extend(labels.cpu().numpy())
                filenames.extend(names)

        return filenames, preds


class EnsemblePredictor:
    """Three-model softmax probability averaging."""

    def __init__(self, m1, m2, m3):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def predict(self, dataloader):
        filenames, preds = [], []

        self.m1.eval()
        self.m2.eval()
        self.m3.eval()

        with torch.no_grad():
            for imgs, names in dataloader:
                imgs = imgs.to(DEVICE)

                out1 = self.m1(imgs)
                out2 = self.m2(imgs)
                out3 = self.m3(imgs)

                p1 = F.softmax(out1, dim=1)
                p2 = F.softmax(out2, dim=1)
                p3 = F.softmax(out3, dim=1)

                avg = (p1 + p2 + p3) / 3.0

                labels = torch.argmax(avg, dim=1)
                preds.extend(labels.cpu().numpy())
                filenames.extend(names)

        return filenames, preds



# Generate and Save CSV
def save_submission(filenames, preds, file_name):
    """Write predictions to a Kaggle-compatible CSV file."""

    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    output_path = os.path.join(SUBMISSION_DIR, file_name)

    with open(output_path, "w") as f:
        f.write("Category,Id\n")
        for i, img_path in enumerate(sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))):
            img_name = os.path.basename(img_path)
            label = full_dataset.classes[preds[i]]
            f.write(f"{label},{img_name}\n")
        f.close()
    print(f"Submission file saved: {output_path}")



# Main Interface Function
def run_inference():
    """Entry point for running single-model or ensemble inference."""


    parser = argparse.ArgumentParser(description="Generate Inference From Cassava Disease Models")
    parser.add_argument("--model", type=str, default="convnext",
                        choices=["convnext", "efficientnet", "densenet"],
                        help="Model name to evaluate")
    parser.add_argument("--ensemble", action="store_true",
                        help="Evaluate 3-model ensemble instead")

    args = parser.parse_args()

    # Load models
    ml = ModelLoader()

    # Ensemble prediction
    if args.ensemble:
        print("Running Ensemble Inference...")
        m1 = ml.load_efficientnet_b4()
        m2 = ml.load_convnext_tiny()
        m3 = ml.load_densenet121()

        predictor = EnsemblePredictor(m1, m2, m3)
        filenames, preds = predictor.predict(submit_test_loader)
        save_submission(filenames, preds, "submission_ensemble.csv")
        return
    else:
        # Single-model prediction
        model_name = args.model
        print(f"Running Single Model: {model_name}")

        if model_name == "efficientnet":
            model = ml.load_efficientnet_b4()
            filename = "submission_efficientnet.csv"

        elif model_name == "convnext":
            model = ml.load_convnext_tiny()
            filename = "submission_convnext.csv"

        elif model_name == "densenet":
            model = ml.load_densenet121()
            filename = "submission_densenet.csv"

        else:
            raise ValueError("Unknown model: choose efficientnet | convnext | densenet")

        predictor = Predictor(model)
        filenames, preds = predictor.predict(submit_test_loader)

        save_submission(filenames, preds, filename)

# Example usage when testing from terminal:example python -m src.inference --model convnext
if __name__ == "__main__":
    run_inference()