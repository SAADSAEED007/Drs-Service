from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drs.detector import BallDetection


@dataclass
class TrackedPoint:
    frame_index: int
    x: float
    y: float
    radius: float
    confidence: float
    interpolated: bool = False


@dataclass
class TrackingSummary:
    tracking_confidence: float
    detected_count: int
    interpolated_count: int
    smoothing_residual: float


class BallTracker:
    def __init__(self) -> None:
        self._max_prediction_gap = 6

    def build_trajectory(self, detections: list[BallDetection | None]) -> list[TrackedPoint]:
        first_detection = next((detection for detection in detections if detection is not None), None)
        if first_detection is None:
            return []

        kalman = self._build_kalman_filter(first_detection)
        trajectory: list[TrackedPoint] = []
        last_radius = first_detection.radius
        missing_run = 0

        for frame_index, detection in enumerate(detections):
            predicted_state = kalman.predict()
            predicted_x = float(predicted_state[0][0])
            predicted_y = float(predicted_state[1][0])

            if detection is not None:
                measurement = np.array([[np.float32(detection.center[0])], [np.float32(detection.center[1])]])
                corrected = kalman.correct(measurement)
                x = float(corrected[0][0])
                y = float(corrected[1][0])
                last_radius = detection.radius
                confidence = min(max(detection.confidence * 0.92 + 0.05, 0.0), 0.99)
                interpolated = False
                missing_run = 0
            elif trajectory and missing_run < self._max_prediction_gap:
                x = predicted_x
                y = predicted_y
                confidence = max(0.12, trajectory[-1].confidence * 0.8)
                interpolated = True
                missing_run += 1
            else:
                continue

            trajectory.append(
                TrackedPoint(
                    frame_index=frame_index,
                    x=x,
                    y=y,
                    radius=float(last_radius),
                    confidence=float(confidence),
                    interpolated=interpolated,
                )
            )

        return self._smooth_trajectory(trajectory)

    def summarize_tracking(
        self,
        detections: list[BallDetection | None],
        trajectory: list[TrackedPoint],
    ) -> TrackingSummary:
        detected_points = [detection for detection in detections if detection is not None]
        detected_count = len(detected_points)
        interpolated_count = sum(1 for point in trajectory if point.interpolated)
        if not trajectory:
            return TrackingSummary(
                tracking_confidence=0.0,
                detected_count=detected_count,
                interpolated_count=interpolated_count,
                smoothing_residual=1.0,
            )

        coverage = len(trajectory) / max(len(detections), 1)
        interpolation_penalty = 1.0 - min(interpolated_count / max(len(trajectory), 1), 0.45)
        average_detection_confidence = (
            float(np.mean([detection.confidence for detection in detected_points])) if detected_points else 0.0
        )
        smoothing_residual = self._calculate_smoothing_residual(detections, trajectory)
        residual_score = max(0.0, 1.0 - smoothing_residual / 20.0)
        monotonicity_score = self._calculate_monotonicity_score(trajectory)
        tracking_confidence = (
            average_detection_confidence * 0.35
            + coverage * 0.2
            + interpolation_penalty * 0.15
            + residual_score * 0.15
            + monotonicity_score * 0.15
        )

        return TrackingSummary(
            tracking_confidence=round(float(max(0.0, min(tracking_confidence, 0.99))), 2),
            detected_count=detected_count,
            interpolated_count=interpolated_count,
            smoothing_residual=round(float(smoothing_residual), 2),
        )

    def _build_kalman_filter(self, first_detection: BallDetection) -> cv2.KalmanFilter:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.12
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman.statePost = np.array(
            [[np.float32(first_detection.center[0])], [np.float32(first_detection.center[1])], [0], [0]],
            np.float32,
        )
        return kalman

    def _smooth_trajectory(self, trajectory: list[TrackedPoint]) -> list[TrackedPoint]:
        if len(trajectory) < 4:
            return trajectory

        frame_indices = np.array([point.frame_index for point in trajectory], dtype=np.float64)
        xs = np.array([point.x for point in trajectory], dtype=np.float64)
        ys = np.array([point.y for point in trajectory], dtype=np.float64)

        xs = self._moving_average(xs, window=5)
        ys = self._moving_average(ys, window=5)

        x_degree = 2 if len(trajectory) >= 6 else 1
        y_degree = 3 if len(trajectory) >= 8 else 2 if len(trajectory) >= 5 else 1
        x_coeff = np.polyfit(frame_indices, xs, x_degree)
        y_coeff = np.polyfit(frame_indices, ys, y_degree)

        smoothed: list[TrackedPoint] = []
        for point in trajectory:
            smoothed.append(
                TrackedPoint(
                    frame_index=point.frame_index,
                    x=float(np.polyval(x_coeff, point.frame_index)),
                    y=float(np.polyval(y_coeff, point.frame_index)),
                    radius=point.radius,
                    confidence=point.confidence,
                    interpolated=point.interpolated,
                )
            )

        return smoothed

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < window:
            return values

        kernel = np.ones(window, dtype=np.float64) / window
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[: len(values)]

    def _calculate_smoothing_residual(
        self,
        detections: list[BallDetection | None],
        trajectory: list[TrackedPoint],
    ) -> float:
        smoothed_by_frame = {point.frame_index: point for point in trajectory}
        residuals: list[float] = []
        for frame_index, detection in enumerate(detections):
            tracked_point = smoothed_by_frame.get(frame_index)
            if detection is None or tracked_point is None:
                continue
            residuals.append(float(np.hypot(detection.center[0] - tracked_point.x, detection.center[1] - tracked_point.y)))

        return float(np.mean(residuals)) if residuals else 999.0

    def _calculate_monotonicity_score(self, trajectory: list[TrackedPoint]) -> float:
        if len(trajectory) < 3:
            return 0.25

        x_deltas = np.diff(np.array([point.x for point in trajectory], dtype=np.float64))
        forward_motion = float(np.mean(x_deltas > -1.5))
        return max(0.0, min(forward_motion, 1.0))
