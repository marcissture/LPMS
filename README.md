# Violence Detection Monitoring System

A comprehensive PyTorch-based monitoring system for detecting violent behavior in surveillance videos.

## Project Overview

This project implements a deep learning solution for real-time violence detection in video streams, with a full GUI application and decision-making system.

### Components

1. **Data Pipeline** (`dataset.py`)
   - Loads video files and annotations
   - Preprocesses frames (resizing, normalization)
   - Creates train/validation/test splits
   - Implements frame sampling and sequence batching

2. **Model Architecture** (`model.py`)
   - CNN-LSTM: ResNet50 feature extractor + LSTM temporal modeling
   - Attention mechanism for temporal attention
   - Dropout and regularization for robustness
   - Multiple model variants available

3. **Training Pipeline** (`train.py`)
   - Full training loop with validation
   - Checkpointing and early stopping
   - Learning rate scheduling
   - Gradient clipping for stability

4. **Evaluation Metrics** (`evaluate.py`)
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curve
   - Confusion Matrix
   - Per-class metrics
   - Visualization generation

5. **Decision Making** (`decision_maker.py`)
   - Rule-based alert system
   - Confidence thresholding
   - Alert cooldown to prevent spam
   - Report generation

6. **GUI Application** (`gui.py`)
   - Main monitoring window with video playback
   - Statistics dashboard
   - Results table with predictions
   - Real-time video processing
   - Model and video file selection

## Project Structure

```
LPMS/
├── data/
│   ├── Videos/              # Video files (Normal_*.mp4, Violent_*.mp4)
│   └── Annotations/         # JSON annotation files
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py            # Model architectures
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation metrics
│   ├── decision_maker.py   # Decision making logic
│   ├── gui.py              # PyQt5 GUI application
│   └── main.py             # Main training script
├── models/                  # Saved model weights
├── checkpoints/             # Training checkpoints
├── logs/                    # Training logs
├── outputs/                 # Generated outputs
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

All configuration is managed in `src/config.py`. Key settings:

- **Model Config**: Input size, sequence length, LSTM parameters
- **Training Config**: Learning rate, batch size, number of epochs
- **Data Config**: Train/val/test split ratios
- **Decision Config**: Confidence thresholds, alert cooldown
- **GUI Config**: Window size, display settings

## Usage

### Training the Model

```bash
cd src
python main.py
```

This will:
1. Load dataset from `data/Videos/` and `data/Annotations/`
2. Create data loaders with train/val/test splits
3. Train the CNN-LSTM model
4. Evaluate on test set
5. Generate visualizations
6. Save trained model to `models/`

### Running the GUI Application

```bash
cd src
python -m PyQt5.api

# Then run the GUI
python gui.py
```

GUI Features:
- Load trained model
- Select video file
- Process video and get predictions
- View statistics and results
- Real-time monitoring display

### Training from Code

```python
from src.dataset import DataManager
from src.train import Trainer
from src.evaluate import Evaluator

# Load data
data_manager = DataManager()
train_loader, val_loader, test_loader = data_manager.create_data_loaders()

# Train model
trainer = Trainer(model_name="cnn_lstm")
history = trainer.train(train_loader, val_loader)

# Evaluate
evaluator = Evaluator(trainer.model)
metrics, predictions, labels, probabilities = evaluator.evaluate(test_loader)
```

## Model Architecture

### CNN-LSTM Model

```
Input (Batch, Sequence_Length, 3, 224, 224)
    ↓
ResNet50 Feature Extractor (for each frame)
    ↓
Features (Batch, Sequence_Length, 2048)
    ↓
LSTM Layer 1 (256 units)
    ↓
LSTM Layer 2 (256 units)
    ↓
Multi-Head Attention (8 heads)
    ↓
FC Layer 1 (128 units, ReLU)
    ↓
FC Layer 2 (num_classes units, softmax)
    ↓
Output (Batch, num_classes)
```

## Key Features

### Data Preparation (45 points)
- Frame extraction and resizing
- Normalization (ImageNet statistics)
- Stratified train/val/test splitting
- Sequence batching
- Frame sampling optimization

### Model Configuration (60 points)
- ResNet50 backbone
- LSTM for temporal modeling
- Multi-head attention mechanism
- Dropout regularization
- Gradient clipping

### Model Training (40 points)
- Full training loop
- Checkpoint saving
- Loss and accuracy tracking
- Early stopping

### Model Evaluation (45 points)
- Multiple metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix
- ROC curve visualization
- Per-class evaluation
- Training history plots

### Model Integration (35 points)
- GUI application
- Real-time video processing
- Decision making system
- Prediction visualization

### Decision Making (15-50 points)
- Rule-based alerts
- Confidence thresholding
- Alert cooldown system
- Action recommendations

### Visualization (20-100+ points)
- PyQt5 GUI with 3+ windows
- Real-time video monitoring
- Statistics dashboard
- Results table
- Training visualizations

## Model Performance

Expected metrics on test set:
- Accuracy: >85%
- Precision: >80%
- Recall: >80%
- F1-Score: >0.80
- ROC-AUC: >0.90

## Decision Making Rules

### Alert Levels:
- **NORMAL** (0): Normal behavior detected
- **CAUTION** (1): Possible violence, low confidence
- **WARNING** (2): Violent behavior, cooldown active
- **CRITICAL** (3): Violent behavior detected, high confidence, action required

### Rules:
1. If prediction=0 (Normal): Continue monitoring
2. If prediction=1 (Violent) AND confidence > threshold AND no cooldown:
   - Trigger CRITICAL alert
   - Log incident
   - Activate cooldown
3. If prediction=1 AND confidence <= threshold:
   - Trigger CAUTION alert
   - Increase vigilance

## Logging

All events are logged to `logs/monitoring_system.log`:
- Model loading/saving
- Training progress
- Evaluation results
- Decision making events
- Errors and warnings

## Project Points Breakdown

| Component | Max Points | Status |
|-----------|-----------|--------|
| Project Description | 20 | To be completed |
| Dataset | 15 | Complete |
| Data Preparation | 45 | Complete |
| Model Configuration | 60 | Complete |
| Model Training | 40 | Complete |
| Model Evaluation | 45 | Complete |
| Model Integration | 35 | Complete |
| Decision Making | 30 | Complete |
| Visualization | 60 | Complete |
| **TOTAL** | **350+** | **~290 implemented** |

## Future Enhancements

- [ ] Multi-GPU support
- [ ] Model quantization
- [ ] ONNX export
- [ ] Web API with Flask/FastAPI
- [ ] Real-time streaming support
- [ ] Mobile app integration
- [ ] Advanced RL-based decision making
- [ ] Multi-camera support

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- ResNet Paper: https://arxiv.org/abs/1512.03385
- LSTM Tutorial: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Attention Mechanism: https://arxiv.org/abs/1706.03762

## License

MIT License

## Author

Your Name

## Acknowledgments

- Dataset: Violent and Normal Behavior Videos Dataset (Kaggle)
- Framework: PyTorch
- GUI: PyQt5
