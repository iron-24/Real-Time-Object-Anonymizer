"""
Object detection module using YOLOv8 with Apple Silicon MPS support.

[INTERVIEW TALKING POINT]: YOLOv8 nano chosen over larger variants (small/medium/large)
for real-time inference speed while maintaining acceptable accuracy for anonymization use case.
The tradeoff: ~2-3% lower mAP but 3-5x faster inference on MPS.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single object detection."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    class_id: int


class AnonymizerDetector:
    """
    YOLOv8-based object detector with automatic MPS/CPU device selection.

    Designed for real-time anonymization: detects faces, license plates, and screens.
    """

    # COCO class IDs we care about (Phase 1: faces only via person detection)
    # Note: YOLO doesn't have explicit "face" class in COCO, we'll use person detection
    # and apply face detection via preprocessing in Phase 1, then fine-tune in Phase 2+
    TARGET_CLASSES = {
        0: "person",  # Will be used for face detection via face-specific models later
        # Future: custom classes for license_plate and screen after fine-tuning
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        device: Optional[str] = None
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to model weights. If None, downloads YOLOv8n pretrained.
            confidence_threshold: Minimum confidence score for detections (0.0-1.0).
            device: Device to use ('mps', 'cpu', or None for auto-detect).
        """
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)

        # Load model (Phase 1: use yolov8n, Phase 2+: load fine-tuned weights)
        if model_path is None:
            model_path = "yolov8n.pt"  # Auto-downloads from Ultralytics

        self.model = YOLO(model_path)

        # Move model to device
        # [INTERVIEW TALKING POINT]: Ultralytics handles device placement internally,
        # but we explicitly verify MPS availability to provide CPU fallback for non-Apple machines
        print(f"[Detector] Loaded {model_path} on device: {self.device}")

    def _setup_device(self, device: Optional[str] = None) -> str:
        """
        Auto-detect best available device: MPS (Apple Silicon) > CPU.

        [INTERVIEW TALKING POINT]: MPS backend provides 2-3x speedup over CPU on M-series chips
        for YOLOv8 inference. Fallback to CPU ensures portability across platforms.

        Args:
            device: Explicitly specified device, or None for auto-detection.

        Returns:
            Device string: 'mps' or 'cpu'.
        """
        if device:
            return device

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"

        return "cpu"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single frame.

        Args:
            frame: Input image as numpy array (BGR format from OpenCV).

        Returns:
            List of Detection objects with bounding boxes and metadata.
        """
        # Run inference (ultralytics handles device placement internally)
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False  # Suppress per-frame logging
        )

        detections = []

        # Parse results (ultralytics returns Results object)
        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # Only process target classes (Phase 1: person for face detection)
                if class_id not in self.TARGET_CLASSES:
                    continue

                # Extract bbox coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                confidence = float(boxes.conf[i].item())
                class_name = self.TARGET_CLASSES[class_id]

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_name=class_name,
                    class_id=class_id
                ))

        return detections

    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold for detections.

        Args:
            threshold: New confidence threshold (0.0-1.0).
        """
        assert 0.0 <= threshold <= 1.0, "Threshold must be between 0.0 and 1.0"
        self.confidence_threshold = threshold
        print(f"[Detector] Updated confidence threshold to {threshold:.2f}")
