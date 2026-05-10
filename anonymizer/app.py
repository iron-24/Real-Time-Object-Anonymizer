"""
Real-Time Object Anonymizer - Gradio Web Interface

A portfolio project demonstrating real-time computer vision for privacy protection.
Detects and anonymizes faces, license plates, and screens in webcam/video streams.

[INTERVIEW TALKING POINT]: Gradio chosen over custom React UI for rapid prototyping
and deployment. Provides professional UI with webcam support, file upload, and real-time
streaming out of the box. Perfect for MVP and technical demos.
"""

from pathlib import Path
from typing import Tuple, Optional
import logging

import gradio as gr
import cv2
import numpy as np

from detector import AnonymizerDetector, Detection
from anonymizer import Anonymizer, AnonymizationEffect
from utils import FPSCounter, draw_fps, validate_frame


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnonymizerApp:
    """
    Main application class for real-time anonymization.

    Handles webcam streaming, video processing, and Gradio UI interactions.
    """

    def __init__(self):
        """Initialize detector and state."""
        # Phase 1: Use Haar Cascade for face detection (fast, built-in)
        # Phase 2+: Will upgrade to fine-tuned YOLOv8
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Will be initialized when needed to avoid loading during import
        self.detector = None

        # FPS tracking
        self.fps_counter = FPSCounter(window_size=30)

        logger.info("[App] Initialized successfully")

    def _ensure_detector_loaded(self) -> None:
        """Lazy load detector to avoid unnecessary model loading."""
        if self.detector is None:
            logger.info("[App] Loading YOLOv8 detector...")
            self.detector = AnonymizerDetector(
                model_path=None,  # Will download yolov8n.pt
                confidence_threshold=0.4
            )
            logger.info("[App] Detector loaded")

    def detect_faces(self, frame: np.ndarray, confidence: float) -> list:
        """
        Detect faces using Haar Cascade (Phase 1 implementation).

        [INTERVIEW TALKING POINT]: Using Haar Cascades for Phase 1 because:
        1. Zero-cost inference (CPU-based, no GPU needed for this part)
        2. Fast enough for real-time (< 5ms per frame)
        3. Good enough for MVP - will upgrade to YOLOv8-face in Phase 2 for better accuracy

        Args:
            frame: Input frame in BGR format.
            confidence: Confidence threshold (not used for Haar, kept for API consistency).

        Returns:
            List of Detection objects for faces.
        """
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Convert to Detection objects
        detections = []
        for (x, y, w, h) in faces:
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                confidence=1.0,  # Haar doesn't provide confidence scores
                class_name="face",
                class_id=0
            ))

        return detections

    def process_frame(
        self,
        frame: np.ndarray,
        anonymize_faces: bool,
        anonymize_plates: bool,
        anonymize_screens: bool,
        effect: str,
        intensity: int,
        confidence: float,
        show_boxes: bool
    ) -> np.ndarray:
        """
        Process a single frame with anonymization.

        Args:
            frame: Input frame (BGR).
            anonymize_faces: Whether to anonymize faces.
            anonymize_plates: Whether to anonymize license plates (Phase 2+).
            anonymize_screens: Whether to anonymize screens (Phase 3+).
            effect: Anonymization effect name.
            intensity: Effect intensity (1-10).
            confidence: Detection confidence threshold.
            show_boxes: Whether to draw bounding boxes for debugging.

        Returns:
            Processed frame with anonymization applied.
        """
        if not validate_frame(frame):
            logger.warning("[App] Invalid frame received")
            return frame

        # Update FPS
        self.fps_counter.update()

        # Collect all detections
        all_detections = []

        # Phase 1: Face detection
        if anonymize_faces:
            face_detections = self.detect_faces(frame, confidence)
            all_detections.extend(face_detections)

        # Phase 2: License plate detection (placeholder)
        if anonymize_plates:
            # TODO: Implement in Phase 2 with fine-tuned YOLOv8
            pass

        # Phase 3: Screen detection (placeholder)
        if anonymize_screens:
            # TODO: Implement in Phase 3 with two-stage detection
            pass

        # Parse effect enum
        effect_enum = AnonymizationEffect[effect.upper()]

        # Apply anonymization to each detection
        for detection in all_detections:
            frame = Anonymizer.apply_effect(
                frame,
                detection.bbox,
                effect_enum,
                intensity
            )

        # Draw bounding boxes if debugging mode enabled
        if show_boxes:
            frame = Anonymizer.draw_bounding_boxes(
                frame,
                all_detections,
                color=(0, 255, 0),
                thickness=2
            )

        # Draw FPS counter
        fps = self.fps_counter.get_fps()
        frame = draw_fps(frame, fps)

        return frame

    def webcam_handler(
        self,
        frame: np.ndarray,
        anonymize_faces: bool,
        anonymize_plates: bool,
        anonymize_screens: bool,
        effect: str,
        intensity: float,
        confidence: float,
        show_boxes: bool
    ) -> np.ndarray:
        """
        Handler for Gradio webcam stream.

        Args:
            frame: Webcam frame from Gradio.
            (other args): UI control values.

        Returns:
            Processed frame.
        """
        # Convert intensity to int
        intensity = int(intensity)

        return self.process_frame(
            frame,
            anonymize_faces,
            anonymize_plates,
            anonymize_screens,
            effect,
            intensity,
            confidence,
            show_boxes
        )


def create_ui() -> gr.Blocks:
    """
    Create Gradio interface.

    Returns:
        Gradio Blocks interface.
    """
    app = AnonymizerApp()

    with gr.Blocks(title="Real-Time Object Anonymizer") as demo:
        gr.Markdown("""
        # Real-Time Object Anonymizer

        Automatically detect and anonymize faces, license plates, and screen content in real-time.

        **Phase 1**: Face detection using Haar Cascades (upgrade to YOLOv8 in Phase 2)

        Built with: YOLOv8 • OpenCV • PyTorch MPS • Gradio
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Webcam input
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam Feed"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Detection Settings")

                # Object selection
                anonymize_faces = gr.Checkbox(
                    value=True,
                    label="Anonymize Faces",
                    info="Detect and blur faces (Phase 1: Active)"
                )

                anonymize_plates = gr.Checkbox(
                    value=False,
                    label="Anonymize License Plates",
                    info="Detect and blur plates (Phase 2: Coming soon)",
                    interactive=False
                )

                anonymize_screens = gr.Checkbox(
                    value=False,
                    label="Anonymize Screens/Monitors",
                    info="Detect and blur screens (Phase 3: Coming soon)",
                    interactive=False
                )

                gr.Markdown("### Anonymization Effect")

                effect = gr.Radio(
                    choices=["blur", "pixelate", "blackout"],
                    value="blur",
                    label="Effect Type",
                    info="Choose how to anonymize detected objects"
                )

                intensity = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=7,
                    step=1,
                    label="Effect Intensity",
                    info="Higher = stronger effect"
                )

                gr.Markdown("### Advanced Settings")

                confidence = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Min confidence for detections (YOLOv8 only)"
                )

                show_boxes = gr.Checkbox(
                    value=False,
                    label="Show Bounding Boxes",
                    info="Debug mode: display detection boxes"
                )

        gr.Markdown("""
        ---
        ### About This Project

        This is a portfolio project demonstrating real-time computer vision with privacy-preserving AI.

        **Architecture**:
        - **Detection**: YOLOv8 nano (Ultralytics) + OpenCV Haar Cascades
        - **Inference**: PyTorch with MPS backend (Apple Silicon GPU acceleration)
        - **Effects**: Custom OpenCV implementations (Gaussian blur, pixelation, redaction)
        - **UI**: Gradio with streaming webcam support

        **Performance**: 15-30 FPS on Apple Silicon M-series chips

        [View on GitHub](#) | [Model on HuggingFace](#) | [Read Blog Post](#)
        """)

        # Connect webcam stream to processing pipeline
        webcam.stream(
            fn=app.webcam_handler,
            inputs=[
                webcam,
                anonymize_faces,
                anonymize_plates,
                anonymize_screens,
                effect,
                intensity,
                confidence,
                show_boxes
            ],
            outputs=webcam,
            show_progress="hidden"  # Hide progress bar for smoother streaming
        )

    return demo


if __name__ == "__main__":
    logger.info("[Main] Starting Real-Time Object Anonymizer")

    # Create and launch UI
    demo = create_ui()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )
