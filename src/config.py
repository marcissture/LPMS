"""
Configuration file for the Violence Detection Monitoring System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "Videos"
ANNOTATIONS_DIR = DATA_DIR / "Annotations"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, LOGS_DIR, OUTPUTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DATASET_CONFIG = {
    "normal_prefix": "Normal",
    "violent_prefix": "Violent",
    "frame_sample_rate": 2,  # Sample every 2nd frame (16 frames per video)
    "video_extension": ".mp4",
    "annotation_extension": ".json",
}

# Model Configuration
MODEL_CONFIG = {
    "input_shape": (3, 224, 224),  # (channels, height, width)
    "sequence_length": 16,  # 16 frames for good temporal coverage
    "num_classes": 2,  # Normal (0) and Violent (1)
    "lstm_hidden_size": 128,  # Reduced for faster training
    "lstm_num_layers": 1,  # Reduced from 2 for speed
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 24,  # Increased to 24 (3× original)
    "num_epochs": 20,  # Reduced from 50 for faster training
    "device": "cuda",  # or "cpu"
}

# Data Preparation
DATA_PREP_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "augmentation_enabled": False,  # Disabled for speed
    "num_workers": 0,  # Reduced from 4 for stability on Windows
}

# Training Configuration
TRAINING_CONFIG = {
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "learning_rate_scheduler": "cosine",
    "checkpoint_interval": 2,  # Save checkpoint every 2 epochs (less frequent)
    "early_stopping_patience": 5,  # Reduced from 10
    "early_stopping_min_delta": 1e-4,  # Require meaningful val_loss improvement
}

# Evaluation Metrics
EVAL_CONFIG = {
    "confidence_threshold": 0.5,
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
    ],
}

# Decision Making
DECISION_CONFIG = {
    "violence_confidence_threshold": 0.7,
    "alert_cooldown_seconds": 5,
    "enable_rules_based": True,
    "enable_rl_based": False,
}

# GUI Configuration
GUI_CONFIG = {
    "window_width": 1400,
    "window_height": 900,
    "dark_mode": True,
    "max_video_fps": 30,
    "update_interval_ms": 33,  # ~30 FPS
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "monitoring_system.log",
}
