"""
Module: evaluate.py
===================

Author: Emmanuel & Tiana
Created: December 2025

Description:
------------
Evaluation utilities for Cassava Leaf Disease Classification models.

This module provides a clean and extensible framework for computing model
performance on validation or test splits. It supports both single-model
evaluation and ensemble evaluation using averaged softmax probabilities.

Features:
---------
- Evaluator class:
    * Computes loss, accuracy, confusion matrix, and classification report.
    * Supports both hard labels and soft-label targets.

- EnsembleEvaluator class:
    * Combines predictions from 3 models using probability averaging.
    * Computes the same set of evaluation metrics as the single evaluator.

- Visualization and Saaving utilities:
    * Confusion matrix heatmap (Seaborn-based).
    * Classification report (Scikit-learn).
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from src.utils import REPORT_DIR

import warnings
warnings.filterwarnings("ignore")



# Base Evaluation
class Evaluator:
    """
    Evaluates a single PyTorch model on a labeled dataset.

    Computes:
    - Cross-entropy loss
    - Accuracy
    - Confusion matrix
    - Classification report
    """

    def __init__(self, device,model_name,class_names=None):
        self.device = device
        self.filename = model_name
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, model, dataloader):
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)

                # Handle soft labels (one-hot targets)
                if labels.ndim == 2:
                    labels = torch.argmax(labels, dim=1)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

        #self._plot_confusion_matrix(all_labels, all_preds)
        #self._print_classification_report(all_labels, all_preds)
        self._save_confusion_and_report(all_labels, all_preds, f"{self.filename}_confusion_matrix_and_classification_report.png") 

        return avg_loss, accuracy

    def _plot_confusion_matrix(self, labels, preds):
        """Displays a confusion matrix heatmap."""
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f"{self.filename} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def _print_classification_report(self, labels, preds):
        """Prints the classification report."""
        print(f"\n{self.filename} Classification Report:")
        print(classification_report(labels, preds, target_names=self.class_names))

    def _save_confusion_and_report(self, labels, preds,filename):
        """
        Save a combined figure containing:
        - Confusion matrix (top)
        - Classification report (bottom)
        """
        cm = confusion_matrix(labels, preds)

        report_str = classification_report(labels, preds, target_names=self.class_names)
        path = os.path.join(REPORT_DIR, filename)

        fig = plt.figure(figsize=(10, 12))

        # Confusion Matrix
        ax1 = fig.add_subplot(2, 1, 1)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax1
        )
        ax1.set_title(f"{self.filename} Confusion Matrix", fontsize=16)
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        # Classification Report
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.axis("off")
        ax2.text(
            0, 0.95, f"{self.filename} Classification Report",
            fontsize=16, fontweight="bold", ha="left", va="top"
        )
        ax2.text(
            0, 0.85, report_str,
            fontsize=12, fontfamily="monospace", ha="left", va="top"
        )

        plt.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved combined confusion matrix and report: {path}")


# Ensemble Evaluation
class EnsembleEvaluator:
    """
    Evaluates an ensemble of 3 models using softmax probability averaging.

    Models:
        m1, m2, m3 (PyTorch models)

    Computes:
        - Averaged probabilities
        - Loss (log-prob cross-entropy)
        - Accuracy
        - Confusion matrix
        - Classification report
    """

    def __init__(self, device, model_name="Ensembled", class_names=None):
        self.device = device
        self.filename = model_name
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, m1, m2, m3, dataloader):

        m1.eval()
        m2.eval()
        m3.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Soft-label handling
                if labels.ndim == 2:
                    labels = torch.argmax(labels, dim=1)

                out1 = m1(images)
                out2 = m2(images)
                out3 = m3(images)

                p1 = F.softmax(out1, dim=1)
                p2 = F.softmax(out2, dim=1)
                p3 = F.softmax(out3, dim=1)

                avg_probs = (p1 + p2 + p3) / 3

                loss = self.criterion(avg_probs.log(), labels)
                total_loss += loss.item()

                preds = torch.argmax(avg_probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Ensemble Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

        # self._plot_confusion_matrix(all_labels, all_preds)
        # self._print_classification_report(all_labels, all_preds)
        self._save_confusion_and_report(all_labels, all_preds, f"{self.filename}_confusion_matrix_and_classification_report.png")


        return avg_loss, accuracy

    def _plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title("Ensemble Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def _print_classification_report(self, labels, preds):
        print("\nEnsemble Classification Report:")
        print(classification_report(labels, preds, target_names=self.class_names))

    def _save_confusion_and_report(self, labels, preds,filename):
        """
        Save a combined figure containing:
        - Confusion matrix (top)
        - Classification report (bottom)
        """
        cm = confusion_matrix(labels, preds)

        report_str = classification_report(labels, preds, target_names=self.class_names)
        path = os.path.join(REPORT_DIR, filename)

        fig = plt.figure(figsize=(10, 12))

        # Confusion Matrix
        ax1 = fig.add_subplot(2, 1, 1)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax1
        )
        ax1.set_title(f"{self.filename} Confusion Matrix", fontsize=16)
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        # Classification Report
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.axis("off")
        ax2.text(
            0, 0.95, f"{self.filename} Classification Report",
            fontsize=16, fontweight="bold", ha="left", va="top"
        )
        ax2.text(
            0, 0.85, report_str,
            fontsize=12, fontfamily="monospace", ha="left", va="top"
        )

        plt.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved combined confusion matrix and report: {path}")



# Terminal Interface
def main():
    from src.utils import DEVICE,AVAILABLE_MODELS,ALL_DATALOADERS_DATASET
    from src.inference import ModelLoader

    full_dataset = ALL_DATALOADERS_DATASET["full_dataset"]
    test_loader = ALL_DATALOADERS_DATASET["test_loader"]

    model = ModelLoader().load_convnext_tiny()

    parser = argparse.ArgumentParser(description="Evaluate Cassava Disease Models")
    parser.add_argument("--model", type=str, default="convnext",
                        choices=["convnext", "efficientnet", "densenet"],
                        help="Model name to evaluate")
    parser.add_argument("--ensemble", action="store_true",
                        help="Evaluate 3-model ensemble instead")

    args = parser.parse_args()

    ml = ModelLoader()

    if args.ensemble:
        print("Running ensemble evaluation...")
        m1 = ml.load_efficientnet_b4()
        m2 = ml.load_convnext_tiny()
        m3 = ml.load_densenet121()

        ensemble_eval = EnsembleEvaluator(device=DEVICE, class_names=full_dataset.classes)
        loss, acc = ensemble_eval.evaluate(m1=m1, m2=m2, m3=m3, dataloader=test_loader)

    else:
        print(f"Evaluating model: {args.model}")
        if args.model == "convnext":
            model = ml.load_convnext_tiny()
        elif args.model == "efficientnet":
            model = ml.load_efficientnet_b4()
        elif args.model == "densenet":
            model = ml.load_densenet121()

        evaluator = Evaluator(device=DEVICE,model_name=AVAILABLE_MODELS[args.model],class_names=full_dataset.classes)

        loss, acc = evaluator.evaluate(model=model, dataloader=test_loader)


# Example usage when testing from terminal:example python -m src.evaluate --model convnext
if __name__ == "__main__":
    main()