"""
Main training script
"""

import logging
import sys
from pathlib import Path

from config import LOGS_DIR, MODELS_DIR, LOGGING_CONFIG
from dataset import DataManager
from train import Trainer
from evaluate import Evaluator

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("Violence Detection Model Training")
    logger.info("=" * 60)

    # Step 1: Load dataset
    logger.info("\n[Step 1] Loading Dataset...")
    data_manager = DataManager()
    dataset_info = data_manager.load_dataset_info()
    logger.info(f"Dataset Info: {dataset_info}")

    # Load annotations
    annotations = data_manager.load_annotations()
    logger.info(f"Annotations loaded: {len(annotations)} files")

    # Step 2: Create data loaders
    logger.info("\n[Step 2] Creating Data Loaders...")
    train_loader, val_loader, test_loader = data_manager.create_data_loaders()

    # Step 3: Initialize trainer
    logger.info("\n[Step 3] Initializing Model...")
    trainer = Trainer(model_name="cnn_lstm", device="cuda")
    logger.info(f"Model: {trainer.model}")
    logger.info(f"Device: {trainer.device}")

    # Step 4: Train model
    logger.info("\n[Step 4] Training Model...")
    history = trainer.train(train_loader, val_loader)

    # Step 5: Save trained model
    logger.info("\n[Step 5] Saving Model...")
    model_save_path = MODELS_DIR / "violence_detection_model.pt"
    trainer.save_model(model_save_path)
    logger.info(f"Model saved to: {model_save_path}")

    # Step 6: Evaluate on test set
    logger.info("\n[Step 6] Evaluating Model on Test Set...")
    evaluator = Evaluator(trainer.model, device=str(trainer.device))
    metrics, predictions, labels, probabilities = evaluator.evaluate(test_loader)

    # Log metrics
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    for metric_name, metric_value in metrics.items():
        if metric_name not in ["confusion_matrix", "precision_per_class", "recall_per_class", "f1_per_class"]:
            logger.info(f"{metric_name}: {metric_value}")

    logger.info("\nConfusion Matrix:")
    for row in metrics["confusion_matrix"]:
        logger.info(str(row))

    # Step 7: Generate visualizations
    logger.info("\n[Step 7] Generating Visualizations...")
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix as sklearn_cm
    cm = sklearn_cm(labels, predictions)
    evaluator.plot_confusion_matrix(cm, save_path=MODELS_DIR / "confusion_matrix.png")

    # Plot ROC curve
    evaluator.plot_roc_curve(labels, probabilities, save_path=MODELS_DIR / "roc_curve.png")

    # Plot training history
    evaluator.plot_training_history(history, save_path=MODELS_DIR / "training_history.png")

    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Completed!")
    logger.info("=" * 60)
    logger.info(f"Model saved: {model_save_path}")
    logger.info(f"Visualizations saved to: {MODELS_DIR}")
    logger.info(f"Logs saved to: {LOGGING_CONFIG['log_file']}")


if __name__ == "__main__":
    main()
