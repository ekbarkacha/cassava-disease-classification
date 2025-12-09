"""
Module: training.py
===================

Author: Emmanuel & Tiana
Created: December 2025

Description:
------------
Training pipeline for Cassava Leaf Disease Classification.

This module provides a clean and extensible framework for
training deep learning models using PyTorch.

Features:
---------
- ModelFactory class for building and configuring pretrained CNN architectures
  including EfficientNet-B4, ConvNeXt-Tiny, and DenseNet121 with custom heads
  for multi-class classification.
- Trainer class implementing a full training loop with:
    * validation per epoch
    * MixUp and CutMix augmentation
    * cosine learning-rate scheduling
    * label smoothing
    * early stopping with best-weight restoration
- Automated saving of training curves.
"""


from __future__ import annotations
import os
import argparse
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from torchvision import models
from torchvision.transforms import v2
from sklearn.metrics import f1_score

from src.evaluate import Evaluator
from src.utils import (NUM_CLASSES, REPORT_DIR, WEIGHTS_DIR, DEVICE, ALL_DATALOADERS_DATASET)

import warnings
warnings.filterwarnings("ignore")

full_dataset = ALL_DATALOADERS_DATASET["full_dataset"]
train_loader = ALL_DATALOADERS_DATASET["train_loader"]
valid_loader = ALL_DATALOADERS_DATASET["valid_loader"]
test_loader = ALL_DATALOADERS_DATASET["test_loader"]


# MODEL FACTORY CLASS
class ModelFactory:
    """
    Factory to create and configure pretrained models for classification.
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes

    def create_models(self) -> Dict[str, nn.Module]:
        """
        Create all configured models and update final classifier layers.
        Returns:
            dict mapping model_name -> nn.Module
        """
        model_dict = {
            "DenseNet121": models.densenet121(pretrained=True),
            "EfficientNet-B4": models.efficientnet_b4(pretrained=True),
            "ConvNeXt": models.convnext_tiny(pretrained=True),
        }

        # Update classifier heads
        model_dict["EfficientNet-B4"].classifier[1] = nn.Linear(
            model_dict["EfficientNet-B4"].classifier[1].in_features, self.num_classes
        )
        model_dict["ConvNeXt"].classifier[-1] = nn.Linear(
            model_dict["ConvNeXt"].classifier[-1].in_features, self.num_classes
        )
        model_dict["DenseNet121"].classifier = nn.Linear(
            model_dict["DenseNet121"].classifier.in_features, self.num_classes
        )

        return model_dict


# TRAINER CLASS
class Trainer:
    """
    Handles the full training pipeline including:
    - training loop
    - validation loop
    - mixup / cutmix augmentation
    - schedulers
    - early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_on: str = "epoch",
        use_mixup: bool = False,
        early_stopping: bool = False,
        patience: int = 3,
        model_name: str = "Model"
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.scheduler_on = scheduler_on
        self.use_mixup = use_mixup
        self.early_stopping = early_stopping
        self.patience = patience
        self.model_name = model_name

        self.val_criterion = nn.CrossEntropyLoss()

        # MixUp / CutMix setup
        if use_mixup:
            cutmix = v2.CutMix(num_classes=NUM_CLASSES)
            mixup = v2.MixUp(num_classes=NUM_CLASSES)
            self.mix_transform = v2.RandomChoice([cutmix, mixup])
        else:
            self.mix_transform = None

        self.model.to(device)

    #  TRAINING LOOP
    def train(self, num_epochs: int) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Trains the model for the given number of epochs.

        Returns:
            (model, history_dict)
        """

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies, val_f1_scores = [], [], []

        best_f1 = 0
        wait = 0
        best_model_wts = None

        for epoch in range(num_epochs):

            # TRAIN
            self.model.train()
            running_train_loss, correct_train, total_train = 0.0, 0, 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # MixUp / CutMix augmentation
                if self.mix_transform:
                    images, labels = self.mix_transform(images, labels)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(labels, dim=1) if labels.ndim == 2 else labels

                correct_train += (predicted == true_labels).sum().item()
                total_train += labels.size(0)

            avg_train_loss = running_train_loss / len(self.train_loader)
            train_accuracy = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            # VALIDATION
            avg_val_loss, val_accuracy, val_f1 = self.validate()
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)

            # SCHEDULER
            if self.scheduler:
                if self.scheduler_on == "val_loss":
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # Print progress
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            #  EARLY STOPPING
            if self.early_stopping:
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    wait = 0
                    best_model_wts = self.model.state_dict()
                else:
                    wait += 1
                    if wait >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                        break

        # Load best weights if needed
        if best_model_wts:
            self.model.load_state_dict(best_model_wts)

        # Save Training curves
        self.plot_loss_accuracy(
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            filename=f"{self.model_name}_training_curve.png",
            save_dir=REPORT_DIR
        )

        history = {
            "train_loss": train_losses,
            "train_acc": train_accuracies,
            "val_loss": val_losses,
            "val_acc": val_accuracies,
            "val_f1": val_f1_scores,
        }

        return self.model, history

    #  VALIDATION
    def validate(self) -> Tuple[float, float, float]:
        """
        Runs one full validation epoch.

        Returns:
            avg_val_loss, val_accuracy, val_f1
        """
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                loss = self.val_criterion(outputs, labels)
                running_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(labels, dim=1) if labels.ndim == 2 else labels

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())

                correct += (predicted == true_labels).sum().item()
                total += labels.size(0)

        avg_val_loss = running_loss / len(self.valid_loader)
        val_accuracy = correct / total
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_val_loss, val_accuracy, val_f1
    
    def plot_loss_accuracy(
        self,    
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        filename,
        save_dir: str = REPORT_DIR
    ):
        """
        Save loss and accuracy plots for training and validation.
        """

        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(12, 5))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss vs Epochs ({self.model_name})')
        plt.legend()

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Epochs ({self.model_name})')
        plt.legend()

        plt.tight_layout()

        # SAVE PLOTS
        path = os.path.join(save_dir, filename)

        plt.savefig(path, dpi=300, bbox_inches="tight")

        plt.close()

        print(f"Saved: {path}")


def parse_arguments():
    """
    Parses command line arguments for training control.
    """
    parser = argparse.ArgumentParser(
        description="Train Cassava models using the modular Trainer class."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "DenseNet121", "EfficientNet-B4", "ConvNeXt"],
        help="Specify which models to train. Use 'all' or list model names.",
    )

    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Enable MixUp/CutMix augmentation."
    )

    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on validation F1 score."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default number of epochs for all models."
    )

    return parser.parse_args()

#   TRAINING EXECUTION
if __name__ == "__main__":

    args = parse_arguments()

    model_factory = ModelFactory(num_classes=NUM_CLASSES)
    model_dict = model_factory.create_models()

    # Determine models to train
    if args.models == ["all"]:
        selected_models = model_dict.items()
    else:
        selected_models = [(m, model_dict[m]) for m in args.models]

    histories = {}
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for model_name, model in selected_models:
        print(f"\n=== Training {model_name} ===\n")

        # Loss function
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        if model_name == "EfficientNet-B4":
            if args.epochs:
                epochs = args.epochs
            else:
                epochs = 10
            lr = 0.001
        elif model_name == "ConvNeXt":
            if args.epochs:
                epochs = args.epochs
            else:
                epochs = 10
            lr = 0.0001
        else:
            if args.epochs:
                epochs = args.epochs
            else:
                epochs = 15
            lr = 0.0001

        # print("model_name:",model_name)
        # print("lr:",lr)
        # print("use_mixup:",args.mixup)
        # print("early_stopping:",args.early_stop)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_on="epoch",
            device=DEVICE,
            use_mixup=args.mixup,
            early_stopping=args.early_stop,
            model_name=model_name,
        )

        trained_model, history = trainer.train(num_epochs=epochs)

        # Save model
        save_path = f"{WEIGHTS_DIR}/{model_name.replace(' ', '_')}.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"Saved {model_name}: {save_path}")

        # Test evaluation
        evaluator = Evaluator(device=DEVICE, model_name=model_name, class_names=full_dataset.classes)
        test_loss, test_accuracy = evaluator.evaluate(trained_model, test_loader)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")

        histories[model_name] = history