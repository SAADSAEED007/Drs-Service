from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


@dataclass
class BallDetection:
    center: tuple[float, float]
    radius: float
    confidence: float
    method: str = "cv"
    bbox: tuple[int, int, int, int] | None = None


class BallDetector:
    def __init__(self, yolo_model_path: str | None = None, models_directory: str | Path | None = None) -> None:
        self._background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=180,
            varThreshold=14,
            detectShadows=False,
        )
        self._previous_gray: np.ndarray | None = None
        self._yolo_model = self._load_yolo_model(yolo_model_path=yolo_model_path, models_directory=models_directory)

    def detect(
        self,
        frame: np.ndarray,
        previous_center: tuple[float, float] | None = None,
    ) -> BallDetection | None:
        enhanced = self._prepare_frame(frame)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        candidates: list[BallDetection] = []
        yolo_detection = self._detect_with_yolo(enhanced, previous_center)
        if yolo_detection is not None:
            candidates.append(yolo_detection)

        flow_detection = self._detect_with_optical_flow(gray, previous_center)
        if flow_detection is not None:
            candidates.append(flow_detection)

        cv_detection = self._detect_with_cv(enhanced, previous_center)
        if cv_detection is not None:
            candidates.append(cv_detection)

        self._previous_gray = gray
        if not candidates:
            return None

        return max(candidates, key=self._score_detection)

    def _load_yolo_model(self, yolo_model_path: str | None, models_directory: str | Path | None):
        if YOLO is None:
            return None

        models_root = Path(models_directory) if models_directory is not None else Path("models")
        candidate_paths = [
            yolo_model_path or "",
            str(models_root / "cricket-ball.pt"),
            str(models_root / "yolov8n.pt"),
        ]
        for candidate in candidate_paths:
            if not candidate:
                continue
            model_path = Path(candidate)
            if not model_path.exists():
                continue
            try:
                return YOLO(str(model_path))
            except Exception:
                continue
        return None

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    def _detect_with_yolo(
        self,
        frame: np.ndarray,
        previous_center: tuple[float, float] | None,
    ) -> BallDetection | None:
        if self._yolo_model is None:
            return None

        try:
            results = self._yolo_model.predict(frame, verbose=False, conf=0.08, imgsz=960)
        except Exception:
            return None

        best_detection: BallDetection | None = None
        best_score = float("-inf")
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
                width = max(x2 - x1, 1)
                height = max(y2 - y1, 1)
                radius = max((width + height) / 4.0, 2.0)
                center = (x1 + width / 2.0, y1 + height / 2.0)
                confidence = float(box.conf[0].item()) if hasattr(box, "conf") else 0.2
                score = confidence * 2.0
                if previous_center is not None:
                    score -= float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) * 0.005)

                if score > best_score:
                    best_score = score
                    best_detection = BallDetection(
                        center=(float(center[0]), float(center[1])),
                        radius=float(radius),
                        confidence=float(min(max(confidence, 0.0), 1.0)),
                        method="yolo",
                        bbox=(x1, y1, x2, y2),
                    )

        return best_detection

    def _detect_with_optical_flow(
        self,
        gray: np.ndarray,
        previous_center: tuple[float, float] | None,
    ) -> BallDetection | None:
        if self._previous_gray is None or previous_center is None:
            return None

        previous_point = np.array([[previous_center]], dtype=np.float32)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            previous_point,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None or int(status[0][0]) == 0:
            return None

        x, y = next_points[0][0]
        flow_error = float(error[0][0]) if error is not None else 12.0
        confidence = max(0.15, min(0.78, 1.0 - flow_error / 30.0))
        return BallDetection(
            center=(float(x), float(y)),
            radius=7.0,
            confidence=float(confidence),
            method="optical_flow",
        )

    def _detect_with_cv(
        self,
        frame: np.ndarray,
        previous_center: tuple[float, float] | None,
    ) -> BallDetection | None:
        motion_mask = self._build_motion_mask(frame)
        color_mask = self._build_color_mask(frame)
        edge_mask = self._build_edge_mask(frame)
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(motion_mask, color_mask), edge_mask)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_detection: BallDetection | None = None
        best_score = float("-inf")

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10 or area > 2200:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = float((4 * np.pi * area) / (perimeter * perimeter))
            if circularity < 0.22:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 3 or max(w, h) > 45:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            score = circularity * 115 + min(area / 30.0, 12.0)
            elongation_penalty = abs(w - h) * 0.6
            score -= elongation_penalty
            if previous_center is not None:
                score -= float(np.hypot(cx - previous_center[0], cy - previous_center[1]) * 0.045)

            if score > best_score:
                best_score = score
                best_detection = BallDetection(
                    center=(float(cx), float(cy)),
                    radius=float(max(radius, 2.0)),
                    confidence=float(max(0.0, min(score / 120.0, 0.92))),
                    method="cv",
                    bbox=(x, y, x + w, y + h),
                )

        return best_detection or self._detect_with_hough(frame, previous_center)

    def _build_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        foreground = self._background_subtractor.apply(blurred)
        _, thresholded = cv2.threshold(foreground, 210, 255, cv2.THRESH_BINARY)
        return thresholded

    def _build_color_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 65, 60), (14, 255, 255)) | cv2.inRange(
            hsv, (166, 65, 60), (179, 255, 255)
        )
        white_mask = cv2.inRange(hsv, (0, 0, 170), (179, 70, 255))
        return cv2.bitwise_or(red_mask, white_mask)

    def _build_edge_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    def _detect_with_hough(
        self,
        frame: np.ndarray,
        previous_center: tuple[float, float] | None,
    ) -> BallDetection | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=18,
            param1=90,
            param2=11,
            minRadius=2,
            maxRadius=20,
        )
        if circles is None:
            return None

        best_detection: BallDetection | None = None
        best_score = float("-inf")
        for x, y, radius in circles[0]:
            score = 0.65
            if previous_center is not None:
                score -= float(np.hypot(x - previous_center[0], y - previous_center[1]) * 0.01)
            if score > best_score:
                best_score = score
                best_detection = BallDetection(
                    center=(float(x), float(y)),
                    radius=float(radius),
                    confidence=float(max(0.1, min(score, 0.78))),
                    method="hough",
                )

        return best_detection

    def _score_detection(self, detection: BallDetection) -> float:
        base = detection.confidence
        method_bonus = {
            "yolo": 0.32,
            "optical_flow": 0.16,
            "cv": 0.12,
            "hough": 0.08,
        }.get(detection.method, 0.0)
        return base + method_bonus
