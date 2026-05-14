"""
Decision-making module for the monitoring system
"""

import logging
import time
from enum import Enum
from typing import Tuple, Dict

from config import DECISION_CONFIG

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""

    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


class DecisionMaker:
    """Makes decisions based on model predictions"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_alert_time = 0
        self.alert_cooldown = DECISION_CONFIG["alert_cooldown_seconds"]

    def make_decision(self, prediction: int, confidence: float) -> Tuple[AlertLevel, str, Dict]:
        """
        Make decision based on model prediction
        
        Args:
            prediction: Model prediction (0=Normal, 1=Violent)
            confidence: Model confidence score (0-1)
        
        Returns:
            alert_level: Severity level
            message: Decision message
            action_info: Dictionary with action details
        """
        action_info = {
            "prediction": prediction,
            "confidence": float(confidence),
            "action_taken": False,
            "alert_triggered": False,
            "recommendation": "",
        }

        # Normal behavior
        if prediction == 0:
            alert_level = AlertLevel.NORMAL
            message = f"Normal behavior detected (confidence: {confidence:.2%})"
            action_info["recommendation"] = "Continue monitoring"

        # Violent behavior
        else:
            if confidence >= DECISION_CONFIG["violence_confidence_threshold"]:
                # High confidence violent behavior
                if self._check_alert_cooldown():
                    alert_level = AlertLevel.CRITICAL
                    message = f"ALERT: Violent behavior detected with high confidence ({confidence:.2%})"
                    action_info["action_taken"] = True
                    action_info["alert_triggered"] = True
                    action_info["recommendation"] = "Immediate action required"
                    self._update_alert_time()
                else:
                    alert_level = AlertLevel.WARNING
                    message = f"Violent behavior detected ({confidence:.2%}) - cooldown active"
                    action_info["recommendation"] = "Alert on cooldown"

            else:
                # Low confidence violent behavior
                alert_level = AlertLevel.CAUTION
                message = f"Possible violent behavior detected ({confidence:.2%})"
                action_info["recommendation"] = "Increase vigilance"

        self.logger.info(f"{message}")
        return alert_level, message, action_info

    def _check_alert_cooldown(self) -> bool:
        """Check if alert cooldown has expired"""
        current_time = time.time()
        return current_time - self.last_alert_time >= self.alert_cooldown

    def _update_alert_time(self):
        """Update last alert time"""
        self.last_alert_time = time.time()

    def generate_report(self, predictions, confidences, frame_indices=None) -> Dict:
        """
        Generate report from multiple predictions
        
        Args:
            predictions: List of predictions
            confidences: List of confidence scores
            frame_indices: List of frame indices (optional)
        
        Returns:
            report: Dictionary with analysis
        """
        predictions = list(predictions)
        confidences = list(confidences)

        normal_count = sum(1 for p in predictions if p == 0)
        violent_count = sum(1 for p in predictions if p == 1)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Find violent segments
        violent_segments = []
        current_segment_start = None
        for i, pred in enumerate(predictions):
            if pred == 1:
                if current_segment_start is None:
                    current_segment_start = i
            else:
                if current_segment_start is not None:
                    violent_segments.append((current_segment_start, i - 1))
                    current_segment_start = None

        if current_segment_start is not None:
            violent_segments.append((current_segment_start, len(predictions) - 1))

        report = {
            "total_frames": len(predictions),
            "normal_frames": int(normal_count),
            "violent_frames": int(violent_count),
            "normal_percentage": 100 * normal_count / len(predictions),
            "violent_percentage": 100 * violent_count / len(predictions),
            "average_confidence": float(avg_confidence),
            "violent_segments": violent_segments,
            "num_violent_segments": len(violent_segments),
        }

        return report
