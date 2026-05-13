"""
Utility functions for frame processing and performance monitoring.
"""

import time
from typing import Optional
from collections import deque

import cv2
import numpy as np


class FPSCounter:
    """
    Tracks frames per second over a rolling window.

    Used to monitor real-time performance and display in UI.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over.
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self) -> None:
        """Record timestamp for current frame."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

    def get_fps(self) -> float:
        """
        Calculate current FPS.

        Returns:
            Average FPS over the window, or 0.0 if no frames recorded.
        """
        if not self.frame_times:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)

        if avg_frame_time == 0:
            return 0.0

        return 1.0 / avg_frame_time

    def reset(self) -> None:
        """Reset FPS counter."""
        self.frame_times.clear()
        self.last_time = time.time()


def draw_fps(frame: np.ndarray, fps: float, position: tuple = (10, 30)) -> np.ndarray:
    """
    Draw FPS counter on frame.

    Args:
        frame: Input frame to draw on.
        fps: Current FPS value.
        position: (x, y) position for text.

    Returns:
        Frame with FPS text drawn.
    """
    text = f"FPS: {fps:.1f}"

    # Draw background rectangle for better visibility
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = position
    cv2.rectangle(
        frame,
        (x - 5, y - text_size[1] - 5),
        (x + text_size[0] + 5, y + 5),
        (0, 0, 0),
        -1
    )

    # Draw FPS text
    color = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return frame


def resize_frame(
    frame: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None
) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.

    Args:
        frame: Input frame.
        target_width: Target width (maintains aspect ratio if height not specified).
        target_height: Target height (maintains aspect ratio if width not specified).

    Returns:
        Resized frame.
    """
    h, w = frame.shape[:2]

    if target_width and not target_height:
        # Calculate height to maintain aspect ratio
        scale = target_width / w
        target_height = int(h * scale)
    elif target_height and not target_width:
        # Calculate width to maintain aspect ratio
        scale = target_height / h
        target_width = int(w * scale)
    elif not target_width and not target_height:
        # No resize needed
        return frame

    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized


def validate_frame(frame: np.ndarray) -> bool:
    """
    Validate that frame is a valid image array.

    Args:
        frame: Input frame to validate.

    Returns:
        True if frame is valid, False otherwise.
    """
    if frame is None:
        return False

    if not isinstance(frame, np.ndarray):
        return False

    if len(frame.shape) not in [2, 3]:
        return False

    if frame.size == 0:
        return False

    return True
