"""
PyQt5 GUI Application for Violence Detection Monitoring System
"""

import sys
import logging
import numpy as np
from pathlib import Path
import torch
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QMessageBox,
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QTextEdit

from config import GUI_CONFIG, MODEL_CONFIG
from model import get_model
from decision_maker import DecisionMaker, AlertLevel

logger = logging.getLogger(__name__)


class VideoProcessor(QThread):
    """Thread for processing videos"""

    progress = pyqtSignal(int)
    frame_processed = pyqtSignal(np.ndarray, int, float)
    processing_finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model, video_path, device):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.device = device
        self.is_running = True

    def run(self):
        """Process video"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            predictions = []
            confidences = []
            frame_buffer = []
            frame_count = 0

            self.model.eval()

            with torch.no_grad():
                while self.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Prepare frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    frame_tensor = torch.from_numpy(frame_resized).float().to(self.device)
                    frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0

                    frame_buffer.append(frame_tensor)

                    # Process when buffer is full
                    if len(frame_buffer) == MODEL_CONFIG["sequence_length"]:
                        sequence = torch.stack(frame_buffer).unsqueeze(0)
                        logits = self.model(sequence)
                        probs = torch.softmax(logits, dim=1)

                        pred = logits.argmax(dim=1).item()
                        conf = probs[0, pred].item()

                        predictions.append(pred)
                        confidences.append(conf)

                        self.frame_processed.emit(frame, pred, conf)
                        frame_buffer.pop(0)

                    frame_count += 1
                    progress = int(100 * frame_count / total_frames)
                    self.progress.emit(progress)

            cap.release()

            # Emit results
            report = {
                "total_frames": frame_count,
                "predictions": predictions,
                "confidences": confidences,
                "fps": fps,
            }
            self.processing_finished.emit(report)

        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Error processing video: {e}")


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decision_maker = DecisionMaker()
        self.video_processor = None
        self.current_video_path = None
        self.init_ui()
        self.setup_logging()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Violence Detection Monitoring System")
        self.setGeometry(100, 100, GUI_CONFIG["window_width"], GUI_CONFIG["window_height"])

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Controls
        left_panel = QVBoxLayout()

        # Title
        title = QLabel("Violence Detection System")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        left_panel.addWidget(title)

        # Load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        left_panel.addWidget(self.load_model_btn)

        # Load video button
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_video_btn.setEnabled(False)
        left_panel.addWidget(self.load_video_btn)

        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        left_panel.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Status: Ready")
        left_panel.addWidget(self.status_label)

        left_panel.addStretch()

        # Right panel - Tabs
        tab_widget = QTabWidget()

        # Tab 1: Monitor
        self.monitor_tab = QWidget()
        monitor_layout = QVBoxLayout()
        self.video_display = QLabel("No video loaded")
        self.video_display.setMinimumHeight(400)
        self.video_display.setAlignment(Qt.AlignCenter)
        monitor_layout.addWidget(self.video_display)
        self.monitor_tab.setLayout(monitor_layout)
        tab_widget.addTab(self.monitor_tab, "Monitor")

        # Tab 2: Statistics
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        self.stats_tab.setLayout(stats_layout)
        tab_widget.addTab(self.stats_tab, "Statistics")

        # Tab 3: Results
        self.results_tab = QWidget()
        results_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Frame", "Prediction", "Confidence"])
        results_layout.addWidget(self.results_table)
        self.results_tab.setLayout(results_layout)
        tab_widget.addTab(self.results_tab, "Results")

        # Add to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(tab_widget, 2)

    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def load_model(self):
        """Load model from file"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Models (*.pt *.pth)"
        )

        if model_path:
            try:
                self.model = get_model(
                    "cnn_lstm",
                    num_classes=MODEL_CONFIG["num_classes"],
                    lstm_hidden_size=MODEL_CONFIG["lstm_hidden_size"],
                    lstm_num_layers=MODEL_CONFIG["lstm_num_layers"],
                    dropout=MODEL_CONFIG["dropout"],
                ).to(self.device)

                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()

                self.status_label.setText(f"Status: Model loaded - {Path(model_path).name}")
                self.load_video_btn.setEnabled(True)
                QMessageBox.information(self, "Success", "Model loaded successfully!")
                logger.info(f"Model loaded: {model_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
                logger.error(f"Error loading model: {e}")

    def load_video(self):
        """Load video file"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )

        if video_path:
            self.current_video_path = video_path
            self.status_label.setText(f"Status: Video loaded - {Path(video_path).name}")
            self.process_btn.setEnabled(True)
            logger.info(f"Video loaded: {video_path}")

    def process_video(self):
        """Process video"""
        if not self.model or not self.current_video_path:
            QMessageBox.warning(self, "Error", "Please load both model and video first!")
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.video_processor = VideoProcessor(self.model, self.current_video_path, self.device)
        self.video_processor.progress.connect(self.update_progress)
        self.video_processor.frame_processed.connect(self.display_frame)
        self.video_processor.processing_finished.connect(self.processing_complete)
        self.video_processor.error.connect(self.processing_error)
        self.video_processor.start()

        self.status_label.setText("Status: Processing video...")

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def display_frame(self, frame, prediction, confidence):
        """Display processed frame"""
        # Convert frame to display format
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Add text overlay
        scaled_pixmap = pixmap.scaledToHeight(400)
        self.video_display.setPixmap(scaled_pixmap)

    def processing_complete(self, report):
        """Handle processing completion"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

        # Display statistics
        stats_text = f"""
        === Processing Results ===
        Total Frames: {report['total_frames']}
        Predictions Made: {len(report['predictions'])}
        
        Normal Frames: {sum(1 for p in report['predictions'] if p == 0)}
        Violent Frames: {sum(1 for p in report['predictions'] if p == 1)}
        
        Average Confidence: {np.mean(report['confidences']):.2%}
        """

        self.stats_text.setText(stats_text)

        # Display results in table
        self.results_table.setRowCount(len(report["predictions"]))
        for i, (pred, conf) in enumerate(
            zip(report["predictions"], report["confidences"])
        ):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i)))
            pred_text = "Violent" if pred == 1 else "Normal"
            self.results_table.setItem(i, 1, QTableWidgetItem(pred_text))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{conf:.2%}"))

        self.status_label.setText("Status: Processing complete!")
        QMessageBox.information(self, "Success", "Video processing completed!")
        logger.info("Video processing completed!")

    def processing_error(self, error):
        """Handle processing error"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Processing error: {error}")
        self.status_label.setText("Status: Error occurred!")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
