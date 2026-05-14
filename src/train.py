"""
Training module for violence detection model
"""

import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import json
from typing import Dict, Tuple
import numpy as np

from config import CHECKPOINT_DIR, MODEL_CONFIG, TRAINING_CONFIG
from model import get_model

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training and checkpointing"""

    def __init__(self, model_name="cnn_lstm", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = get_model(model_name, **MODEL_CONFIG).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=MODEL_CONFIG["learning_rate"])
        self.criterion = nn.CrossEntropyLoss()
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(frames)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                self.logger.info(
                    f"Batch {batch_idx + 1}: Loss={loss.item():.4f}, "
                    f"Acc={100 * correct / total:.2f}%"
                )

        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        return avg_loss, avg_acc

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(frames)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        avg_acc = 100 * correct / total
        return avg_loss, avg_acc

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            checkpoint_path,
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint["epoch"]

    def train(self, train_loader, val_loader, num_epochs=None):
        """Full training loop"""
        num_epochs = num_epochs or MODEL_CONFIG["num_epochs"]

        self.logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%"
            )

            # Save checkpoint
            if (epoch + 1) % TRAINING_CONFIG["checkpoint_interval"] == 0:
                self.save_checkpoint(epoch + 1)

            # Early stopping
            min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 0.0)
            if (self.best_val_loss - val_loss) > min_delta:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint("best")
            else:
                self.early_stopping_counter += 1
                if (
                    self.early_stopping_counter
                    >= TRAINING_CONFIG["early_stopping_patience"]
                ):
                    self.logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience: {TRAINING_CONFIG['early_stopping_patience']})"
                    )
                    break

        self.logger.info("Training completed!")
        return self.history

    def save_model(self, model_path):
        """Save trained model"""
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved: {model_path}")

    def load_model(self, model_path):
        """Load trained model"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.logger.info(f"Model loaded: {model_path}")
