from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drs.tracker import TrackedPoint


@dataclass
class ImpactResult:
    frame_index: int
    point_px: tuple[int, int]
    point_norm: tuple[float, float]
    confidence: float
    method: str


class ImpactDetector:
    def detect(self, frames: list[np.ndarray], ball_trajectory: list[TrackedPoint]) -> ImpactResult:
        if not frames or not ball_trajectory:
            raise ValueError("Impact detection requires frames and a ball trajectory")

        frame_height, frame_width = frames[0].shape[:2]
        trajectory_by_frame = {point.frame_index: point for point in ball_trajectory}

        velocity_candidate = self._detect_velocity_change(ball_trajectory)
        if velocity_candidate is not None:
            return self._build_result(
                trajectory_by_frame.get(velocity_candidate, ball_trajectory[-1]),
                frame_width,
                frame_height,
                0.78,
                "velocity_change",
            )

        motion_stop_candidate = self._detect_motion_stop(ball_trajectory)
        if motion_stop_candidate is not None:
            contour_candidate = self._confirm_pad_contour(frames, ball_trajectory, motion_stop_candidate)
            if contour_candidate is not None:
                return self._build_result(
                    trajectory_by_frame.get(contour_candidate, ball_trajectory[-1]),
                    frame_width,
                    frame_height,
                    0.72,
                    "contour_pad",
                )
            return self._build_result(
                trajectory_by_frame.get(motion_stop_candidate, ball_trajectory[-1]),
                frame_width,
                frame_height,
                0.62,
                "motion_stop",
            )

        contour_only_candidate = self._confirm_pad_contour(frames, ball_trajectory, ball_trajectory[-1].frame_index)
        if contour_only_candidate is not None:
            return self._build_result(
                trajectory_by_frame.get(contour_only_candidate, ball_trajectory[-1]),
                frame_width,
                frame_height,
                0.58,
                "contour_pad",
            )

        return self._build_result(ball_trajectory[-1], frame_width, frame_height, 0.3, "fallback")

    def _detect_velocity_change(self, trajectory: list[TrackedPoint]) -> int | None:
        if len(trajectory) < 3:
            return None

        velocities: list[np.ndarray] = []
        for previous, current in zip(trajectory[:-1], trajectory[1:]):
            frame_delta = max(current.frame_index - previous.frame_index, 1)
            velocities.append(
                np.array(
                    [
                        (current.x - previous.x) / frame_delta,
                        (current.y - previous.y) / frame_delta,
                    ],
                    dtype=np.float64,
                )
            )

        for index in range(1, len(velocities)):
            prev_velocity = velocities[index - 1]
            current_velocity = velocities[index]
            prev_speed = float(np.linalg.norm(prev_velocity))
            current_speed = float(np.linalg.norm(current_velocity))
            if prev_speed <= 1e-6:
                continue

            drop_ratio = 1.0 - (current_speed / prev_speed)
            denominator = max(prev_speed * current_speed, 1e-6)
            cos_theta = float(np.clip(np.dot(prev_velocity, current_velocity) / denominator, -1.0, 1.0))
            direction_change = float(np.degrees(np.arccos(cos_theta)))
            if drop_ratio > 0.4 and direction_change > 25.0:
                return trajectory[index].frame_index

        return None

    def _detect_motion_stop(self, trajectory: list[TrackedPoint]) -> int | None:
        if len(trajectory) < 4:
            return None

        speeds: list[float] = []
        for previous, current in zip(trajectory[:-1], trajectory[1:]):
            frame_delta = max(current.frame_index - previous.frame_index, 1)
            speeds.append(float(np.hypot(current.x - previous.x, current.y - previous.y) / frame_delta))

        for index in range(1, len(speeds) - 1):
            if speeds[index - 1] > 2.5 and speeds[index] < 1.0 and speeds[index + 1] < 1.0:
                return trajectory[index + 1].frame_index

        return None

    def _confirm_pad_contour(
        self,
        frames: list[np.ndarray],
        trajectory: list[TrackedPoint],
        candidate_frame_index: int,
    ) -> int | None:
        frame_lookup = {point.frame_index: point for point in trajectory}
        start_index = max(candidate_frame_index - 5, 0)
        end_index = min(candidate_frame_index + 5, len(frames) - 1)

        for frame_index in range(start_index, end_index + 1):
            tracked_point = frame_lookup.get(frame_index)
            if tracked_point is None:
                continue

            frame = frames[frame_index]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, (0, 20, 70), (25, 180, 255))
            white_mask = cv2.inRange(hsv, (0, 0, 170), (179, 55, 255))
            contour_mask = cv2.morphologyEx(
                cv2.bitwise_or(skin_mask, white_mask),
                cv2.MORPH_CLOSE,
                np.ones((9, 9), np.uint8),
            )
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) <= 3000:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if x - 30 <= tracked_point.x <= x + w + 30 and y - 30 <= tracked_point.y <= y + h + 30:
                    return frame_index

        return None

    def _build_result(
        self,
        tracked_point: TrackedPoint,
        frame_width: int,
        frame_height: int,
        confidence: float,
        method: str,
    ) -> ImpactResult:
        point_px = (int(round(tracked_point.x)), int(round(tracked_point.y)))
        return ImpactResult(
            frame_index=tracked_point.frame_index,
            point_px=point_px,
            point_norm=(
                float(max(0.0, min(point_px[0] / max(frame_width, 1), 1.0))),
                float(max(0.0, min(point_px[1] / max(frame_height, 1), 1.0))),
            ),
            confidence=float(confidence),
            method=method,
        )
