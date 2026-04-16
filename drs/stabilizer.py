from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StabilizationResult:
    frames: list[np.ndarray]
    confidence: float
    method: str


class VideoStabilizer:
    def stabilize(self, frames: list[np.ndarray]) -> StabilizationResult:
        if len(frames) < 3:
            return StabilizationResult(frames=frames, confidence=0.2, method="identity")

        stabilized_frames: list[np.ndarray] = [frames[0]]
        transforms: list[np.ndarray] = []
        previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_points = cv2.goodFeaturesToTrack(
                previous_gray,
                maxCorners=160,
                qualityLevel=0.01,
                minDistance=18,
                blockSize=3,
            )
            if previous_points is None or len(previous_points) < 10:
                transforms.append(np.eye(2, 3, dtype=np.float32))
                stabilized_frames.append(frame)
                previous_gray = current_gray
                continue

            current_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, None)
            if current_points is None or status is None:
                transforms.append(np.eye(2, 3, dtype=np.float32))
                stabilized_frames.append(frame)
                previous_gray = current_gray
                continue

            valid_previous = previous_points[status.flatten() == 1]
            valid_current = current_points[status.flatten() == 1]
            if len(valid_previous) < 8 or len(valid_current) < 8:
                transforms.append(np.eye(2, 3, dtype=np.float32))
                stabilized_frames.append(frame)
                previous_gray = current_gray
                continue

            transform, _ = cv2.estimateAffinePartial2D(valid_current, valid_previous)
            if transform is None:
                transform = np.eye(2, 3, dtype=np.float32)

            transforms.append(transform.astype(np.float32))
            stabilized_frames.append(self._warp_frame(frame, transform))
            previous_gray = current_gray

        confidence = self._estimate_confidence(transforms)
        return StabilizationResult(frames=stabilized_frames, confidence=confidence, method="optical_flow_affine")

    def _warp_frame(self, frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return stabilized

    def _estimate_confidence(self, transforms: list[np.ndarray]) -> float:
        if not transforms:
            return 0.0

        translations = [float(np.hypot(transform[0, 2], transform[1, 2])) for transform in transforms]
        average_translation = float(np.mean(translations)) if translations else 0.0
        return round(float(max(0.2, min(1.0 - average_translation / 20.0, 0.92))), 2)
