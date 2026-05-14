"""
Model architectures for violence detection
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for video classification
    Uses ResNet50 as feature extractor + LSTM for temporal modeling
    """

    def __init__(
        self,
        num_classes=2,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        dropout=0.3,
        pretrained=True,
    ):
        super(CNNLSTMModel, self).__init__()

        # CNN Feature Extractor (ResNet50)
        self.cnn = models.resnet50(pretrained=pretrained)
        # Remove final classification layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        # Freeze ResNet50 backbone to save memory
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn_output_size = 2048

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size, num_heads=8, batch_first=True, dropout=dropout
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, channels, height, width)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_length = x.size(0), x.size(1)

        # Extract CNN features for each frame
        # Reshape: (batch, seq_len, C, H, W) -> (batch*seq_len, C, H, W)
        x = x.view(batch_size * seq_length, x.size(2), x.size(3), x.size(4))
        cnn_features = self.cnn(x)  # (batch*seq_len, 2048, 1, 1)

        # Reshape back: (batch*seq_len, 2048) -> (batch, seq_len, 2048)
        cnn_features = cnn_features.view(batch_size, seq_length, -1)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_features)  # (batch, seq_len, hidden)

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last hidden state
        x = attn_out[:, -1, :]  # (batch, hidden_size)

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class LSTMModel(nn.Module):
    """
    Pure LSTM model with attention for video classification
    Takes pre-extracted features as input
    """

    def __init__(
        self,
        input_size=2048,
        num_classes=2,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        dropout=0.3,
    ):
        super(LSTMModel, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size, num_heads=8, batch_first=True, dropout=dropout
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last hidden state
        x = attn_out[:, -1, :]

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SimpleCNN(nn.Module):
    """
    Simple CNN model for frame-level classification
    """

    def __init__(self, num_classes=2, dropout=0.3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(model_name="cnn_lstm", num_classes=2, **kwargs):
    """
    Factory function to get model
    
    Args:
        model_name: "cnn_lstm", "lstm", or "simple_cnn"
        num_classes: number of output classes
        **kwargs: additional model arguments
    
    Returns:
        model: PyTorch model
    """
    # Filter kwargs to only include relevant parameters for each model
    model_kwargs = {
        "num_classes": num_classes,
    }
    
    # Add model-specific parameters if provided
    if "lstm_hidden_size" in kwargs:
        model_kwargs["lstm_hidden_size"] = kwargs["lstm_hidden_size"]
    if "lstm_num_layers" in kwargs:
        model_kwargs["lstm_num_layers"] = kwargs["lstm_num_layers"]
    if "dropout" in kwargs:
        model_kwargs["dropout"] = kwargs["dropout"]
    
    if model_name == "cnn_lstm":
        return CNNLSTMModel(**model_kwargs, pretrained=True)
    elif model_name == "lstm":
        return LSTMModel(**model_kwargs)
    elif model_name == "simple_cnn":
        return SimpleCNN(**model_kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
