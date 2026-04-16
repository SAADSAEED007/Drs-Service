from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drs.calibrator import CalibrationResult
from drs.tracker import TrackedPoint


@dataclass
class PredictionResult:
    predicted_points: list[tuple[int, int]]
    impact_point: tuple[int, int]
    impact_frame_index: int
    stumps_x: int
    will_hit_stumps: bool
    collision_point: tuple[int, int]
    wickets_top: int
    wickets_bottom: int
    wicket_overlap_ratio: float
    projection_confidence: float
    release_point: tuple[int, int]
    bounce_point: tuple[int, int]


class TrajectoryPredictor:
    def predict(
        self,
        trajectory: list[TrackedPoint],
        calibration: CalibrationResult,
        frame_width: int,
        frame_height: int,
        impact_frame_index: int,
        impact_point: tuple[int, int],
    ) -> PredictionResult:
        if not trajectory:
            raise ValueError("Cannot predict trajectory without tracked points")

        usable_trajectory = [point for point in trajectory if point.frame_index <= impact_frame_index]
        if len(usable_trajectory) < 2:
            usable_trajectory = trajectory[-2:] if len(trajectory) >= 2 else trajectory

        xs = np.array([point.x for point in usable_trajectory], dtype=np.float64)
        ys = np.array([point.y for point in usable_trajectory], dtype=np.float64)
        degree = 3 if len(usable_trajectory) >= 8 else 2 if len(usable_trajectory) >= 5 else 1
        coeff = np.polyfit(xs, ys, degree)

        stumps_x = int(calibration.stumps_px[0])
        start_x = int(round(xs[-1]))
        if stumps_x <= start_x:
            stumps_x = min(frame_width - 1, start_x + max(32, int(frame_width * 0.09)))

        predicted_points: list[tuple[int, int]] = []
        x_values = np.linspace(start_x, stumps_x, 28)
        for x_value in x_values:
            y_value = float(np.polyval(coeff, x_value))
            predicted_points.append((int(round(x_value)), int(round(self._clamp(y_value, 0, frame_height - 1)))))

        collision_point = predicted_points[-1]
        wickets_top = int(calibration.wicket_top_px)
        wickets_bottom = int(calibration.wicket_bottom_px)
        wicket_overlap_ratio = self._calculate_overlap_ratio(collision_point[1], predicted_points, wickets_top, wickets_bottom)
        will_hit_stumps = wicket_overlap_ratio > 0.5
        projection_confidence = self._estimate_projection_confidence(usable_trajectory, frame_width, stumps_x)

        release_point = (int(round(usable_trajectory[0].x)), int(round(usable_trajectory[0].y)))
        bounce_point = self._estimate_bounce_point(usable_trajectory)

        return PredictionResult(
            predicted_points=predicted_points,
            impact_point=impact_point,
            impact_frame_index=impact_frame_index,
            stumps_x=stumps_x,
            will_hit_stumps=will_hit_stumps,
            collision_point=collision_point,
            wickets_top=wickets_top,
            wickets_bottom=wickets_bottom,
            wicket_overlap_ratio=round(float(wicket_overlap_ratio), 2),
            projection_confidence=round(float(projection_confidence), 2),
            release_point=release_point,
            bounce_point=bounce_point,
        )

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))

    def _estimate_bounce_point(self, trajectory: list[TrackedPoint]) -> tuple[int, int]:
        if len(trajectory) < 3:
            point = trajectory[len(trajectory) // 2]
            return int(round(point.x)), int(round(point.y))

        ys = np.array([point.y for point in trajectory], dtype=np.float64)
        deltas = np.diff(ys)
        for index in range(1, len(deltas)):
            if deltas[index - 1] > 0 and deltas[index] <= 0:
                point = trajectory[index]
                return int(round(point.x)), int(round(point.y))

        point = max(trajectory, key=lambda item: item.y)
        return int(round(point.x)), int(round(point.y))

    def _calculate_overlap_ratio(
        self,
        collision_y: int,
        predicted_points: list[tuple[int, int]],
        wickets_top: int,
        wickets_bottom: int,
    ) -> float:
        if not predicted_points:
            return 0.0

        band_radius = max(int(abs(predicted_points[-1][1] - predicted_points[0][1]) * 0.14), 7)
        ball_top = collision_y - band_radius
        ball_bottom = collision_y + band_radius
        overlap = max(0, min(ball_bottom, wickets_bottom) - max(ball_top, wickets_top))
        return float(overlap / max(ball_bottom - ball_top, 1))

    def _estimate_projection_confidence(
        self,
        trajectory: list[TrackedPoint],
        frame_width: int,
        stumps_x: int,
    ) -> float:
        if len(trajectory) < 2:
            return 0.25

        average_confidence = float(np.mean([point.confidence for point in trajectory]))
        x_span = max(point.x for point in trajectory) - min(point.x for point in trajectory)
        distance_to_stumps = abs(stumps_x - trajectory[-1].x) / max(frame_width, 1)
        curvature_score = min(max(len(trajectory) / 18.0, 0.0), 1.0)
        path_score = min(max(x_span / max(frame_width * 0.18, 1), 0.0), 1.0)
        distance_score = 1.0 - min(distance_to_stumps / 0.35, 1.0)
        return max(
            0.2,
            min(average_confidence * 0.45 + path_score * 0.22 + distance_score * 0.18 + curvature_score * 0.15, 0.97),
        )
