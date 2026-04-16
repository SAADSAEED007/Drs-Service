from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.core.settings import settings
from drs.calibrator import PitchCalibrator
from drs.constants import STUMP_HIT_PROXIMITY_PX
from drs.detector import BallDetector, BallDetection
from drs.impact_detector import ImpactDetector
from drs.lbw_rules import LBWRuleEngine
from drs.predictor import TrajectoryPredictor
from drs.renderer import DrsRenderer
from drs.stabilizer import VideoStabilizer
from drs.tracker import BallTracker, TrackedPoint


@dataclass
class ProcessedDrsResult:
    decision: str
    video_path: Path
    video_url: str
    trajectory: list[tuple[float, float]]
    predicted_trajectory: list[tuple[float, float]]
    raw_trajectory: list[dict[str, float | int]]
    projection_points: list[tuple[float, float]]
    impact_point: tuple[float, float]
    impact_frame_index: int
    pitching: str
    impact: str
    wickets: str
    ultra_edge: str
    confidence: float
    tracking_confidence: float
    decision_confidence: float
    decision_reason: str
    umpires_call: bool
    calibration_method: str
    fps: float
    frame_count: int
    stabilization_method: str


class DrsProcessor:
    def __init__(self, output_directory: Path, public_url_prefix: str = "/videos") -> None:
        self.output_directory = output_directory
        self.public_url_prefix = public_url_prefix.rstrip("/")
        self.detector = BallDetector(
            yolo_model_path=settings.yolo_model_path,
            models_directory=settings.models_directory,
        )
        self.tracker = BallTracker()
        self.stabilizer = VideoStabilizer()
        self.calibrator = PitchCalibrator()
        self.impact_detector = ImpactDetector()
        self.predictor = TrajectoryPredictor()
        self.rule_engine = LBWRuleEngine()
        self.renderer = DrsRenderer(enable_replay_audio=settings.enable_replay_audio)

    def process(self, input_video_path: Path) -> ProcessedDrsResult:
        capture = cv2.VideoCapture(str(input_video_path))
        if not capture.isOpened():
            raise ValueError("Unable to open uploaded video")

        original_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frames: list[np.ndarray] = []

        while True:
            success, frame = capture.read()
            if not success:
                break

            frames.append(frame)

        capture.release()

        if not frames:
            raise ValueError("Uploaded video did not contain readable frames")

        stabilization_method = "disabled"
        if settings.enable_stabilization:
            stabilization_result = self.stabilizer.stabilize(frames)
            frames = stabilization_result.frames
            stabilization_method = stabilization_result.method

        detections: list[BallDetection | None] = []
        previous_center: tuple[float, float] | None = None
        for frame in frames:
            detection = self.detector.detect(frame, previous_center)
            detections.append(detection)
            if detection is not None:
                previous_center = detection.center

        trajectory = self.tracker.build_trajectory(detections)
        if len(trajectory) < 4:
            raise ValueError("Unable to reliably detect and track the ball in the uploaded video")

        frame_height, frame_width = frames[0].shape[:2]
        calibration = self.calibrator.calibrate(frames)
        impact = self.impact_detector.detect(frames, trajectory)
        prediction = self.predictor.predict(
            trajectory=trajectory,
            calibration=calibration,
            frame_width=frame_width,
            frame_height=frame_height,
            impact_frame_index=impact.frame_index,
            impact_point=impact.point_px,
        )
        verdict = self.rule_engine.evaluate(calibration, trajectory, impact, prediction)
        tracking_summary = self.tracker.summarize_tracking(detections, trajectory)
        direct_stump_hit = self._is_direct_stump_hit(trajectory, calibration.stumps_px[0], prediction.wickets_top, prediction.wickets_bottom)

        decision = verdict.decision
        wickets = verdict.wickets
        decision_reason = verdict.decision_reason
        decision_confidence = verdict.confidence
        if direct_stump_hit:
            decision = "OUT"
            wickets = "Hitting"
            decision_reason = "Ball tracked to stump position in the wicket band - direct hit."
            decision_confidence = max(decision_confidence, 0.75)

        display_decision = "NOT OUT" if decision == "NOT OUT" else decision

        output_name = f"processed_drs_video_{uuid4().hex}.mp4"
        output_path = self.output_directory / output_name
        output_fps = max(original_fps / 2.0, 8.0)
        self.renderer.render(
            source_frames=frames,
            trajectory=trajectory,
            prediction=prediction,
            decision_text=display_decision,
            output_path=output_path,
            fps=output_fps,
        )

        return ProcessedDrsResult(
            decision=display_decision,
            video_path=output_path,
            video_url=f"{self.public_url_prefix}/{output_name}",
            trajectory=[(point.x / frame_width, point.y / frame_height) for point in trajectory],
            predicted_trajectory=[(x / frame_width, y / frame_height) for x, y in prediction.predicted_points],
            raw_trajectory=[
                {"frame": point.frame_index, "x": round(float(point.x), 2), "y": round(float(point.y), 2)}
                for point in trajectory
            ],
            projection_points=[(x / frame_width, y / frame_height) for x, y in prediction.predicted_points],
            impact_point=impact.point_norm,
            impact_frame_index=impact.frame_index,
            pitching=verdict.pitching,
            impact=verdict.impact,
            wickets=wickets,
            ultra_edge="No Spike",
            confidence=decision_confidence,
            tracking_confidence=tracking_summary.tracking_confidence,
            decision_confidence=decision_confidence,
            decision_reason=decision_reason,
            umpires_call=False if direct_stump_hit else verdict.umpires_call,
            calibration_method=calibration.method,
            fps=output_fps,
            frame_count=len(frames),
            stabilization_method=stabilization_method,
        )

    def _is_direct_stump_hit(
        self,
        trajectory: list[TrackedPoint],
        stumps_x: int,
        wickets_top: int,
        wickets_bottom: int,
    ) -> bool:
        for point in trajectory:
            if point.confidence < 0.5:
                continue
            reached_stumps = abs(int(point.x) - stumps_x) <= STUMP_HIT_PROXIMITY_PX
            inside_wicket_band = wickets_top <= int(point.y) <= wickets_bottom
            if reached_stumps and inside_wicket_band:
                return True
        return False
