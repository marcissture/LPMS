"""
Model evaluation and metrics calculation
"""

import logging
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
from pathlib import Path

from config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates model performance"""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

    def evaluate(self, test_loader, confidence_threshold=0.5):
        """
        Evaluate model on test set
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()

        all_logits = []
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for frames, labels in test_loader:
                frames = frames.to(self.device)
                logits = self.model(frames)
                probs = torch.softmax(logits, dim=1)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_predictions.append(logits.argmax(dim=1).cpu())
                all_probabilities.append(probs.cpu())

        # Concatenate all batches
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        predictions = torch.cat(all_predictions)
        probabilities = torch.cat(all_probabilities)

        # Calculate metrics
        metrics = self._calculate_metrics(
            predictions.numpy(),
            labels.numpy(),
            probabilities.numpy(),
            confidence_threshold,
        )

        return metrics, predictions.numpy(), labels.numpy(), probabilities.numpy()

    def _calculate_metrics(self, predictions, labels, probabilities, confidence_threshold):
        """Calculate evaluation metrics"""
        metrics = {}

        # Accuracy
        accuracy = np.mean(predictions == labels)
        metrics["accuracy"] = float(accuracy)

        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1_score"] = float(f1)

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(labels, predictions, average=None)
        )
        metrics["precision_per_class"] = precision_per_class.tolist()
        metrics["recall_per_class"] = recall_per_class.tolist()
        metrics["f1_per_class"] = f1_per_class.tolist()

        # ROC-AUC (for binary classification)
        if len(np.unique(labels)) == 2:
            violent_probs = probabilities[:, 1]
            try:
                roc_auc = roc_auc_score(labels, violent_probs)
                metrics["roc_auc"] = float(roc_auc)
            except:
                metrics["roc_auc"] = 0.0

        # True Positives, False Positives, etc.
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

        # TPR and FPR
        if tp + fn > 0:
            metrics["tpr"] = float(tp / (tp + fn))
        if fp + tn > 0:
            metrics["fpr"] = float(fp / (fp + tn))

        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        if "roc_auc" in metrics:
            self.logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ["Normal", "Violent"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            yticklabels=class_names,
            xticklabels=class_names,
        )

        plt.setp(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            self.logger.info(f"Confusion matrix saved: {save_path}")

        plt.close()

    def plot_roc_curve(self, labels, probabilities, save_path=None):
        """Plot ROC curve"""
        if len(np.unique(labels)) != 2:
            self.logger.warning("ROC curve only supports binary classification")
            return

        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            self.logger.info(f"ROC curve saved: {save_path}")

        plt.close()

    def plot_training_history(self, history, save_path=None):
        """Plot training and validation losses"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(history["train_acc"], label="Train Accuracy")
        ax2.plot(history["val_acc"], label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            self.logger.info(f"Training history plot saved: {save_path}")

        plt.close()
