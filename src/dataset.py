"""
Data loading and preprocessing module for violence detection
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

from config import (
    VIDEOS_DIR,
    ANNOTATIONS_DIR,
    DATASET_CONFIG,
    DATA_PREP_CONFIG,
    MODEL_CONFIG,
)

logger = logging.getLogger(__name__)


class ViolenceDataset(Dataset):
    """PyTorch Dataset for violence detection videos"""

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        sequence_length: int = 30,
        transform=None,
        frame_sample_rate: int = 10,
    ):
        """
        Args:
            video_paths: List of paths to video files
            labels: List of labels (0=Normal, 1=Violent)
            sequence_length: Number of frames to sample per video
            transform: Torchvision transforms to apply
            frame_sample_rate: Sample every Nth frame
        """
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        self.frame_sample_rate = frame_sample_rate

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load frames from video
        frames = self._load_video_frames(video_path)

        # Convert to tensor
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        else:
            frames = torch.stack([transforms.ToTensor()(f) for f in frames])

        return frames, torch.tensor(label, dtype=torch.long)

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame
            if frame_count % self.frame_sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

            frame_count += 1

        cap.release()

        # Pad with last frame if necessary
        if len(frames) < self.sequence_length:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames.extend([last_frame] * (self.sequence_length - len(frames)))

        return frames[:self.sequence_length]


class DataManager:
    """Manages dataset loading and preprocessing"""

    def __init__(self):
        self.video_files = []
        self.labels = []
        self.annotation_data = {}
        self.logger = logging.getLogger(__name__)

    def load_dataset_info(self) -> Dict:
        """Load video files and labels from directory structure"""
        self.logger.info("Loading dataset information...")

        videos_dir = Path(VIDEOS_DIR)
        normal_prefix = DATASET_CONFIG["normal_prefix"]
        violent_prefix = DATASET_CONFIG["violent_prefix"]

        # Find all video files
        for video_file in videos_dir.glob("*.mp4"):
            video_name = video_file.stem

            if video_name.startswith(normal_prefix):
                self.video_files.append(str(video_file))
                self.labels.append(0)  # Normal = 0
            elif video_name.startswith(violent_prefix):
                self.video_files.append(str(video_file))
                self.labels.append(1)  # Violent = 1

        self.logger.info(
            f"Found {len(self.video_files)} videos: "
            f"{sum(1 for l in self.labels if l == 0)} Normal, "
            f"{sum(1 for l in self.labels if l == 1)} Violent"
        )

        return {"num_videos": len(self.video_files), "labels_distribution": self.labels}

    def load_annotations(self) -> Dict:
        """Load JSON annotation files"""
        self.logger.info("Loading annotations...")

        annotations_dir = Path(ANNOTATIONS_DIR)
        for anno_file in annotations_dir.glob("*.json"):
            with open(anno_file, "r") as f:
                self.annotation_data[anno_file.stem] = json.load(f)

        self.logger.info(f"Loaded {len(self.annotation_data)} annotation files")
        return self.annotation_data

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders"""
        if not self.video_files:
            self.load_dataset_info()

        self.logger.info("Creating data loaders...")

        # Split data
        indices = np.arange(len(self.video_files))
        labels_array = np.array(self.labels)

        # Train/val/test split
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices,
            labels_array,
            test_size=DATA_PREP_CONFIG["val_split"] + DATA_PREP_CONFIG["test_split"],
            random_state=DATA_PREP_CONFIG["random_seed"],
            stratify=labels_array,
        )

        val_idx, test_idx, _, _ = train_test_split(
            temp_idx,
            temp_labels,
            test_size=DATA_PREP_CONFIG["test_split"]
            / (DATA_PREP_CONFIG["val_split"] + DATA_PREP_CONFIG["test_split"]),
            random_state=DATA_PREP_CONFIG["random_seed"],
            stratify=temp_labels,
        )

        # Create datasets
        train_videos = [self.video_files[i] for i in train_idx]
        train_labels = [self.labels[i] for i in train_idx]

        val_videos = [self.video_files[i] for i in val_idx]
        val_labels = [self.labels[i] for i in val_idx]

        test_videos = [self.video_files[i] for i in test_idx]
        test_labels = [self.labels[i] for i in test_idx]

        # Create transforms
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create datasets
        train_dataset = ViolenceDataset(
            train_videos,
            train_labels,
            sequence_length=MODEL_CONFIG["sequence_length"],
            transform=train_transform,
            frame_sample_rate=DATASET_CONFIG["frame_sample_rate"],
        )

        val_dataset = ViolenceDataset(
            val_videos,
            val_labels,
            sequence_length=MODEL_CONFIG["sequence_length"],
            transform=val_test_transform,
            frame_sample_rate=DATASET_CONFIG["frame_sample_rate"],
        )

        test_dataset = ViolenceDataset(
            test_videos,
            test_labels,
            sequence_length=MODEL_CONFIG["sequence_length"],
            transform=val_test_transform,
            frame_sample_rate=DATASET_CONFIG["frame_sample_rate"],
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            shuffle=True,
            num_workers=DATA_PREP_CONFIG["num_workers"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            shuffle=False,
            num_workers=DATA_PREP_CONFIG["num_workers"],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            shuffle=False,
            num_workers=DATA_PREP_CONFIG["num_workers"],
        )

        self.logger.info(
            f"Created loaders - Train: {len(train_loader)}, "
            f"Val: {len(val_loader)}, Test: {len(test_loader)}"
        )

        return train_loader, val_loader, test_loader
