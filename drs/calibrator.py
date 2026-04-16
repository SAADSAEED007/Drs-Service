from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from drs.constants import PITCH_HALF_WIDTH_M, PITCH_LENGTH_M, STUMP_HEIGHT_M

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    stumps_px: tuple[int, int]
    crease_y_px: int
    pitch_homography: np.ndarray
    confidence: float
    method: str
    wicket_top_px: int
    wicket_bottom_px: int


class PitchCalibrator:
    def calibrate(self, frames: list[np.ndarray]) -> CalibrationResult:
        if not frames:
            raise ValueError("Cannot calibrate an empty frame sequence")

        candidate = self._line_detect(frames[:10])
        if candidate is not None:
            return candidate

        logger.warning("Pitch calibration failed; using fallback heuristic geometry")
        return self._fallback_heuristic(frames[0])

    def _line_detect(self, frames: list[np.ndarray]) -> CalibrationResult | None:
        stump_votes: list[tuple[int, int, int]] = []
        crease_votes: list[int] = []

        for frame in frames:
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=max(width // 7, 40),
                maxLineGap=20,
            )
            if lines is None:
                continue

            horizontal_candidates: list[tuple[int, int]] = []
            vertical_candidates: list[tuple[int, int, int]] = []
            for raw_line in lines[:, 0]:
                x1, y1, x2, y2 = map(int, raw_line)
                dx = x2 - x1
                dy = y2 - y1
                if abs(dy) <= 8 and min(y1, y2) >= int(height * 0.6):
                    horizontal_candidates.append((int((y1 + y2) / 2), abs(dx)))
                if abs(dx) <= 8 and min(x1, x2) >= int(width * 0.55):
                    vertical_candidates.append((int((x1 + x2) / 2), min(y1, y2), max(y1, y2)))

            if horizontal_candidates:
                weighted_y = sum(y * max(length, 1) for y, length in horizontal_candidates) / sum(
                    max(length, 1) for _, length in horizontal_candidates
                )
                crease_votes.append(int(round(weighted_y)))

            if len(vertical_candidates) >= 2:
                vertical_candidates.sort(key=lambda item: item[0])
                clusters: list[list[tuple[int, int, int]]] = [[vertical_candidates[0]]]
                for line in vertical_candidates[1:]:
                    if abs(line[0] - clusters[-1][-1][0]) <= 14:
                        clusters[-1].append(line)
                    else:
                        clusters.append([line])

                best_cluster = max(clusters, key=len)
                if len(best_cluster) >= 2:
                    x = int(round(float(np.mean([line[0] for line in best_cluster]))))
                    top = int(round(float(np.mean([line[1] for line in best_cluster]))))
                    bottom = int(round(float(np.mean([line[2] for line in best_cluster]))))
                    stump_votes.append((x, top, bottom))

        if not stump_votes or not crease_votes:
            return None

        stumps_x = int(round(float(np.median([vote[0] for vote in stump_votes]))))
        wicket_top_px = int(round(float(np.median([vote[1] for vote in stump_votes]))))
        wicket_bottom_px = int(round(float(np.median([vote[2] for vote in stump_votes]))))
        crease_y_px = int(round(float(np.median(crease_votes))))

        frame_height, frame_width = frames[0].shape[:2]
        if wicket_bottom_px <= wicket_top_px or crease_y_px <= wicket_top_px:
            return None

        pitch_left = int(frame_width * 0.18)
        homography = self._build_homography(
            pitch_left=pitch_left,
            stump_x=stumps_x,
            pitch_bottom=min(frame_height - 1, max(crease_y_px + 40, int(frame_height * 0.92))),
            crease_y=crease_y_px,
        )
        confidence = 0.75
        if abs((wicket_bottom_px - wicket_top_px) - frame_height * 0.4) < frame_height * 0.18:
            confidence += 0.1
        confidence = float(min(confidence, 0.9))

        return CalibrationResult(
            stumps_px=(stumps_x, wicket_bottom_px),
            crease_y_px=crease_y_px,
            pitch_homography=homography,
            confidence=confidence,
            method="line_detect",
            wicket_top_px=wicket_top_px,
            wicket_bottom_px=wicket_bottom_px,
        )

    def _fallback_heuristic(self, frame: np.ndarray) -> CalibrationResult:
        frame_height, frame_width = frame.shape[:2]
        stumps_x = int(frame_width * 0.82)
        wicket_top = int(frame_height * 0.33)
        wicket_bottom = int(frame_height * 0.73)
        crease_y = int(frame_height * 0.80)
        logger.warning(
            "Using fallback calibration. Frame: %sx%s. Stumps at x=%s, wicket y=%s to %s",
            frame_width,
            frame_height,
            stumps_x,
            wicket_top,
            wicket_bottom,
        )
        homography = self._build_homography(
            pitch_left=int(frame_width * 0.15),
            stump_x=stumps_x,
            pitch_bottom=int(frame_height * 0.92),
            crease_y=crease_y,
        )
        return CalibrationResult(
            stumps_px=(stumps_x, wicket_bottom),
            crease_y_px=crease_y,
            pitch_homography=homography,
            confidence=0.25,
            method="fallback_heuristic",
            wicket_top_px=wicket_top,
            wicket_bottom_px=wicket_bottom,
        )

    def _build_homography(self, pitch_left: int, stump_x: int, pitch_bottom: int, crease_y: int) -> np.ndarray:
        src = np.array(
            [
                [float(pitch_left), float(pitch_bottom)],
                [float(stump_x), float(pitch_bottom)],
                [float(pitch_left), float(crease_y)],
                [float(stump_x), float(crease_y)],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [
                [-PITCH_HALF_WIDTH_M, 0.0],
                [0.0, 0.0],
                [-PITCH_HALF_WIDTH_M, PITCH_LENGTH_M],
                [0.0, PITCH_LENGTH_M],
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(src, dst).astype(np.float64)

    def to_pitch_space(
        self,
        point_px: tuple[float, float],
        calibration: CalibrationResult,
    ) -> tuple[float, float]:
        points = np.array([[[float(point_px[0]), float(point_px[1])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, calibration.pitch_homography)[0, 0]
        return float(transformed[0]), float(transformed[1])

    def estimate_stump_height_px(self, calibration: CalibrationResult) -> float:
        pixel_height = max(calibration.wicket_bottom_px - calibration.wicket_top_px, 1)
        metres_per_pixel = STUMP_HEIGHT_M / float(pixel_height)
        return 1.0 / max(metres_per_pixel, 1e-6)
