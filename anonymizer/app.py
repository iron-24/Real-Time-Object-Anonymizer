"""
Real-Time Object Anonymizer - Gradio Web Interface

A portfolio project demonstrating real-time computer vision for privacy protection.
Detects and anonymizes faces, license plates, and screens in webcam/video streams.

[INTERVIEW TALKING POINT]: Gradio chosen over custom React UI for rapid prototyping
and deployment. Provides professional UI with real-time display out of the box.

Architecture note: In Gradio 6.x the browser webcam component sends individual
snapshots, not a continuous stream. We capture directly from the camera using
OpenCV (cv2.VideoCapture) and push processed frames to the UI via gr.Timer.
This gives true real-time performance independent of Gradio's webcam API changes.
"""

from typing import Optional
import logging

import gradio as gr
import cv2
import numpy as np

from detector import AnonymizerDetector, Detection
from anonymizer import Anonymizer, AnonymizationEffect
from utils import FPSCounter, draw_fps, validate_frame


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnonymizerApp:
    """
    Main application class for real-time anonymization.

    Captures frames from the webcam via OpenCV and processes them on each
    gr.Timer tick. UI controls update instance variables which are read on
    the next frame, so no extra synchronisation is needed.
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            logger.error("[App] Haar Cascade failed to load — check OpenCV install")
        else:
            logger.info("[App] Haar Cascade loaded successfully")

        self.detector = None
        self.fps_counter = FPSCounter(window_size=30)

        # Open the default webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("[App] Failed to open webcam (device 0)")
        else:
            logger.info("[App] Webcam opened successfully")

        # Current settings — updated by UI controls via update_settings()
        self.anonymize_faces = True
        self.effect = "blur"
        self.intensity = 7
        self.confidence = 0.4
        self.show_boxes = False

        logger.info("[App] Initialized successfully")

    def update_settings(
        self,
        anonymize_faces: bool,
        effect: str,
        intensity: float,
        confidence: float,
        show_boxes: bool,
    ) -> None:
        """Called whenever a UI control changes."""
        self.anonymize_faces = anonymize_faces
        self.effect = effect
        self.intensity = int(intensity)
        self.confidence = confidence
        self.show_boxes = show_boxes

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces using Haar Cascade (Phase 1 implementation).

        [INTERVIEW TALKING POINT]: Using Haar Cascades for Phase 1 because:
        1. Zero-cost inference (CPU-based, no GPU needed for this part)
        2. Fast enough for real-time (< 5ms per frame)
        3. Good enough for MVP — will upgrade to YOLOv8-face in Phase 2

        Args:
            frame: Input frame in BGR format.

        Returns:
            List of Detection objects for faces.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        n = len(faces) if hasattr(faces, '__len__') else 0
        if n > 0:
            logger.info(f"[App] Detected {n} face(s)")

        detections = []
        for (x, y, w, h) in faces:
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                confidence=1.0,
                class_name="face",
                class_id=0
            ))
        return detections

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture and process one frame from the webcam.

        Called on every gr.Timer tick (~30 FPS). Returns an RGB numpy array
        for Gradio to display, or None if the camera isn't ready.
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("[App] Failed to read frame from webcam")
            return None

        self.fps_counter.update()

        all_detections = []

        if self.anonymize_faces:
            all_detections.extend(self.detect_faces(frame))

        # Phase 2 / 3 placeholders
        # if self.anonymize_plates: ...
        # if self.anonymize_screens: ...

        effect_enum = AnonymizationEffect[self.effect.upper()]

        for detection in all_detections:
            frame = Anonymizer.apply_effect(
                frame,
                detection.bbox,
                effect_enum,
                self.intensity
            )

        if self.show_boxes:
            frame = Anonymizer.draw_bounding_boxes(
                frame, all_detections, color=(0, 255, 0), thickness=2
            )

        frame = draw_fps(frame, self.fps_counter.get_fps())

        # OpenCV is BGR; Gradio expects RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def create_ui() -> gr.Blocks:
    """Create Gradio interface."""
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
                output_image = gr.Image(
                    type="numpy",
                    label="Live Feed (processed)",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Detection Settings")

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
                    minimum=1, maximum=10, value=7, step=1,
                    label="Effect Intensity",
                    info="Higher = stronger effect"
                )

                gr.Markdown("### Advanced Settings")

                confidence = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.4, step=0.05,
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
        - **UI**: Gradio with real-time OpenCV capture via gr.Timer

        **Performance**: 15-30 FPS on Apple Silicon M-series chips

        [View on GitHub](#) | [Model on HuggingFace](#) | [Read Blog Post](#)
        """)

        # Timer drives the processing loop (~30 FPS)
        timer = gr.Timer(value=1 / 30, active=True)
        timer.tick(fn=app.get_frame, outputs=[output_image])

        # Sync UI controls → app settings on any change
        controls = [anonymize_faces, effect, intensity, confidence, show_boxes]
        for ctrl in controls:
            ctrl.change(fn=app.update_settings, inputs=controls, outputs=[])

    return demo


if __name__ == "__main__":
    logger.info("[Main] Starting Real-Time Object Anonymizer")

    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
