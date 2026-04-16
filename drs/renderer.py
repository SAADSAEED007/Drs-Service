from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from drs.predictor import PredictionResult
from drs.replay import BroadcastReplayDirector, ReplayInstruction
from drs.tracker import TrackedPoint

logger = logging.getLogger(__name__)


class DrsRenderer:
    def __init__(self, enable_replay_audio: bool = True) -> None:
        self._replay_director = BroadcastReplayDirector()
        self._enable_replay_audio = enable_replay_audio

    def render(
        self,
        source_frames: list[np.ndarray],
        trajectory: list[TrackedPoint],
        prediction: PredictionResult,
        decision_text: str,
        output_path: Path,
        fps: float,
    ) -> None:
        if not source_frames:
            raise ValueError("No frames available to render")

        base_height, base_width = source_frames[0].shape[:2]
        target_width, target_height = self._target_output_size(base_width, base_height)
        scale_x = target_width / max(base_width, 1)
        scale_y = target_height / max(base_height, 1)

        scaled_frames = [cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR) for frame in source_frames]
        scaled_trajectory = self._scale_trajectory(trajectory, scale_x, scale_y)
        scaled_prediction = self._scale_prediction(prediction, scale_x, scale_y)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        instructions = self._replay_director.build(
            frame_count=len(scaled_frames),
            impact_frame_index=scaled_prediction.impact_frame_index,
            collision_point=scaled_prediction.collision_point,
        )
        rendered_frames: list[np.ndarray] = []

        trajectory_by_frame = {point.frame_index: point for point in scaled_trajectory}
        revealed_path: list[tuple[int, int]] = []
        current_ball_point: tuple[int, int] | None = None
        current_ball_radius = 9

        for instruction in instructions:
            frame = scaled_frames[instruction.frame_index].copy()
            if instruction.zoom_center is not None and instruction.zoom_scale > 1.0:
                frame = self._zoom_frame(frame, instruction.zoom_center, instruction.zoom_scale)

            point = trajectory_by_frame.get(instruction.frame_index)
            if point is not None:
                current_ball_point = (int(round(point.x)), int(round(point.y)))
                current_ball_radius = max(int(round(point.radius)), 7)
                revealed_path.append(current_ball_point)

            self._draw_background_tint(frame, instruction.stage)
            self._draw_broadcast_guides(frame, scaled_prediction)
            self._draw_score_strip(frame, decision_text, instruction.stage)

            if len(revealed_path) >= 2:
                self._draw_glow_polyline(frame, revealed_path, glow_color=(24, 174, 255), line_color=(90, 230, 255))

            if current_ball_point is not None:
                self._draw_ball_marker(frame, current_ball_point, current_ball_radius)

            if instruction.stage in {"tracking", "impact_zoom", "projection", "decision"}:
                self._draw_bounce_marker(frame, scaled_prediction.bounce_point)
                self._draw_impact_marker(frame, scaled_prediction.impact_point)
                self._draw_glow_dotted_line(frame, scaled_prediction.predicted_points, color=(0, 220, 255), thickness=3, dot_spacing=11)
                self._draw_lbw_projection(frame, scaled_prediction)
                self._draw_release_marker(frame, scaled_prediction.release_point)
                self._draw_virtual_wickets(frame, scaled_prediction)

            if instruction.stage == "decision_pending":
                self._draw_decision_pending(frame)
            elif instruction.stage == "impact_zoom":
                self._draw_focus_banner(frame, "IMPACT ZOOM")
            elif instruction.stage == "projection":
                self._draw_focus_banner(frame, "HAWK-EYE PROJECTION")
            elif instruction.stage == "decision":
                self._draw_final_decision(frame, decision_text)
            else:
                self._draw_tracking_status(frame, scaled_prediction.will_hit_stumps)

            rendered_frames.append(frame)

        self._write_replay(
            frames=rendered_frames,
            output_path=output_path,
            fps=max(fps, 1.0),
            frame_size=(target_width, target_height),
        )

    def _target_output_size(self, width: int, height: int) -> tuple[int, int]:
        target_height = max(720, height)
        scale = target_height / max(height, 1)
        target_width = int(round(width * scale))
        target_width += target_width % 2
        target_height += target_height % 2
        return target_width, target_height

    def _scale_trajectory(self, trajectory: list[TrackedPoint], scale_x: float, scale_y: float) -> list[TrackedPoint]:
        return [
            TrackedPoint(
                frame_index=point.frame_index,
                x=point.x * scale_x,
                y=point.y * scale_y,
                radius=point.radius * ((scale_x + scale_y) / 2.0),
                confidence=point.confidence,
                interpolated=point.interpolated,
            )
            for point in trajectory
        ]

    def _scale_prediction(self, prediction: PredictionResult, scale_x: float, scale_y: float) -> PredictionResult:
        def scale_point(point: tuple[int, int]) -> tuple[int, int]:
            return int(round(point[0] * scale_x)), int(round(point[1] * scale_y))

        return PredictionResult(
            predicted_points=[scale_point(point) for point in prediction.predicted_points],
            impact_point=scale_point(prediction.impact_point),
            impact_frame_index=prediction.impact_frame_index,
            stumps_x=int(round(prediction.stumps_x * scale_x)),
            will_hit_stumps=prediction.will_hit_stumps,
            collision_point=scale_point(prediction.collision_point),
            wickets_top=int(round(prediction.wickets_top * scale_y)),
            wickets_bottom=int(round(prediction.wickets_bottom * scale_y)),
            wicket_overlap_ratio=prediction.wicket_overlap_ratio,
            projection_confidence=prediction.projection_confidence,
            release_point=scale_point(prediction.release_point),
            bounce_point=scale_point(prediction.bounce_point),
        )

    def _zoom_frame(self, frame: np.ndarray, center: tuple[int, int], zoom_scale: float) -> np.ndarray:
        if zoom_scale <= 1.0:
            return frame

        height, width = frame.shape[:2]
        crop_width = int(width / zoom_scale)
        crop_height = int(height / zoom_scale)
        center_x = int(np.clip(center[0], crop_width // 2, width - crop_width // 2))
        center_y = int(np.clip(center[1], crop_height // 2, height - crop_height // 2))
        x1 = max(center_x - crop_width // 2, 0)
        y1 = max(center_y - crop_height // 2, 0)
        x2 = min(x1 + crop_width, width)
        y2 = min(y1 + crop_height, height)
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def _write_replay(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        temporary_path = output_path.with_name(f"{output_path.stem}_raw.mp4")
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        if not writer.isOpened():
            raise ValueError("Unable to open replay video writer")

        for frame in frames:
            writer.write(frame)
        writer.release()

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            logger.warning("ffmpeg not found; replay will use raw mp4v output and may not play in browsers")
            temporary_path.replace(output_path)
            return

        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(temporary_path),
                    "-vcodec",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if self._enable_replay_audio:
                self._add_replay_audio(ffmpeg_path, output_path, fps)
        except subprocess.CalledProcessError as error:
            logger.warning("ffmpeg re-encode failed; falling back to raw mp4v output: %s", error.stderr.strip())
            temporary_path.replace(output_path)
            return
        finally:
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    def _add_replay_audio(self, ffmpeg_path: str, output_path: Path, fps: float) -> None:
        audio_mux_path = output_path.with_name(f"{output_path.stem}_audio.mp4")
        replay_duration = max(3.0, 1.0 / max(fps, 1.0))

        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(output_path),
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency=880:duration={replay_duration}:sample_rate=44100",
                    "-filter_complex",
                    (
                        "[1:a]volume=0.03,afade=t=out:st="
                        f"{max(replay_duration - 0.5, 0.1)}:d=0.5[a1]"
                    ),
                    "-map",
                    "0:v:0",
                    "-map",
                    "[a1]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(audio_mux_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            audio_mux_path.replace(output_path)
        except subprocess.CalledProcessError as error:
            logger.warning("Replay audio mux failed; continuing without sound: %s", error.stderr.strip())
            if audio_mux_path.exists():
                audio_mux_path.unlink(missing_ok=True)

    def _draw_background_tint(self, frame: np.ndarray, stage: str) -> None:
        overlay = frame.copy()
        colors = {
            "decision_pending": (12, 18, 30),
            "tracking": (14, 18, 18),
            "impact_zoom": (24, 20, 16),
            "projection": (10, 20, 28),
            "decision": (8, 10, 12),
        }
        color = colors.get(stage, (15, 15, 15))
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, thickness=-1)
        alpha = 0.12 if stage != "decision" else 0.24
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_broadcast_guides(self, frame: np.ndarray, prediction: PredictionResult) -> None:
        height, width = frame.shape[:2]
        pitch_left = int(width * 0.10)
        pitch_right = int(width * 0.91)
        pitch_top = int(height * 0.24)
        pitch_bottom = int(height * 0.84)
        center_x = int((pitch_left + pitch_right) / 2)

        cv2.line(frame, (pitch_left, pitch_bottom), (pitch_right, pitch_bottom), (225, 225, 225), 2, cv2.LINE_AA)
        cv2.line(frame, (center_x, pitch_top), (center_x, pitch_bottom), (185, 185, 185), 1, cv2.LINE_AA)
        cv2.line(frame, (pitch_left, int(height * 0.58)), (pitch_right, int(height * 0.58)), (110, 160, 200), 1, cv2.LINE_AA)

    def _draw_score_strip(self, frame: np.ndarray, decision_text: str, stage: str) -> None:
        cv2.rectangle(frame, (28, 24), (394, 112), (16, 18, 24), thickness=-1)
        cv2.rectangle(frame, (28, 24), (394, 112), (60, 74, 92), thickness=1)
        cv2.putText(frame, "KHELAAO DRS", (44, 58), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BALL TRACKING + HAWK-EYE", (44, 90), cv2.FONT_HERSHEY_DUPLEX, 0.58, (198, 215, 235), 1, cv2.LINE_AA)

        stage_text = {
            "decision_pending": "DECISION PENDING",
            "tracking": "SLOW MOTION TRACKING",
            "impact_zoom": "PAD / IMPACT CHECK",
            "projection": "PROJECTED PATH",
            "decision": f"FINAL CALL: {decision_text}",
        }.get(stage, "DRS REVIEW")
        text_width = cv2.getTextSize(stage_text, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)[0][0]
        right_box_x1 = frame.shape[1] - max(text_width + 54, 280)
        cv2.rectangle(frame, (right_box_x1, 28), (frame.shape[1] - 28, 78), (8, 12, 18), thickness=-1)
        cv2.putText(frame, stage_text, (right_box_x1 + 18, 62), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 214, 255), 2, cv2.LINE_AA)

    def _draw_tracking_status(self, frame: np.ndarray, will_hit_stumps: bool) -> None:
        label = "HITTING" if will_hit_stumps else "MISSING"
        color = (0, 72, 255) if will_hit_stumps else (0, 190, 90)
        cv2.rectangle(frame, (34, 132), (262, 186), (18, 20, 20), thickness=-1)
        cv2.putText(frame, "TRACK STATUS", (48, 154), cv2.FONT_HERSHEY_DUPLEX, 0.54, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (48, 177), cv2.FONT_HERSHEY_DUPLEX, 0.86, color, 2, cv2.LINE_AA)

    def _draw_decision_pending(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        center = (width // 2, int(height * 0.5))
        for radius, alpha in ((132, 0.10), (92, 0.16), (54, 0.24)):
            overlay = frame.copy()
            cv2.circle(overlay, center, radius, (0, 162, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, "DECISION PENDING", (center[0] - 196, center[1] + 8), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "Synchronising ball tracking and impact analysis", (center[0] - 212, center[1] + 48), cv2.FONT_HERSHEY_DUPLEX, 0.58, (202, 216, 232), 1, cv2.LINE_AA)

    def _draw_focus_banner(self, frame: np.ndarray, label: str) -> None:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (width // 2 - 170, height - 86), (width // 2 + 170, height - 38), (16, 18, 20), thickness=-1)
        cv2.putText(frame, label, (width // 2 - 142, height - 52), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 212, 102), 2, cv2.LINE_AA)

    def _draw_final_decision(self, frame: np.ndarray, decision_text: str) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (10, 10, 12), thickness=-1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        if decision_text == "OUT":
            color = (0, 72, 255)
        elif decision_text == "UMPIRES CALL":
            color = (0, 215, 255)
        else:
            color = (0, 190, 90)

        text_size = cv2.getTextSize(decision_text, cv2.FONT_HERSHEY_DUPLEX, 2.6, 6)[0]
        origin = ((frame.shape[1] - text_size[0]) // 2, int(frame.shape[0] * 0.52))
        cv2.putText(frame, decision_text, origin, cv2.FONT_HERSHEY_DUPLEX, 2.6, color, 6, cv2.LINE_AA)
        cv2.putText(frame, "TV REPLAY DECISION", (origin[0] + 30, origin[1] + 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_glow_polyline(
        self,
        frame: np.ndarray,
        points: list[tuple[int, int]],
        glow_color: tuple[int, int, int],
        line_color: tuple[int, int, int],
    ) -> None:
        if len(points) < 2:
            return

        path = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [path], False, glow_color, 10, cv2.LINE_AA)
        cv2.polylines(frame, [path], False, line_color, 4, cv2.LINE_AA)

    def _draw_ball_marker(self, frame: np.ndarray, center: tuple[int, int], radius: int) -> None:
        cv2.circle(frame, center, radius + 12, (28, 144, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, radius + 5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, max(radius, 8), (36, 78, 255), thickness=-1, lineType=cv2.LINE_AA)

    def _draw_release_marker(self, frame: np.ndarray, point: tuple[int, int]) -> None:
        cv2.circle(frame, point, 10, (114, 255, 180), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "RELEASE", (point[0] + 14, point[1] - 12), cv2.FONT_HERSHEY_DUPLEX, 0.54, (114, 255, 180), 1, cv2.LINE_AA)

    def _draw_bounce_marker(self, frame: np.ndarray, point: tuple[int, int]) -> None:
        cv2.ellipse(frame, point, (18, 8), 0, 0, 360, (0, 182, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BOUNCE", (point[0] - 22, point[1] - 12), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 182, 255), 1, cv2.LINE_AA)

    def _draw_impact_marker(self, frame: np.ndarray, impact_point: tuple[int, int]) -> None:
        cv2.drawMarker(frame, impact_point, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=3)
        cv2.putText(frame, "IMPACT", (impact_point[0] + 14, impact_point[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    def _draw_virtual_wickets(self, frame: np.ndarray, prediction: PredictionResult) -> None:
        stump_offsets = (-12, 0, 12)
        for offset in stump_offsets:
            x = prediction.stumps_x + offset
            cv2.line(frame, (x, prediction.wickets_top), (x, prediction.wickets_bottom), (255, 255, 255), 4, cv2.LINE_AA)

        bail_y = prediction.wickets_top + 10
        cv2.line(frame, (prediction.stumps_x - 18, bail_y), (prediction.stumps_x - 3, bail_y), (220, 220, 220), 2, cv2.LINE_AA)
        cv2.line(frame, (prediction.stumps_x + 3, bail_y), (prediction.stumps_x + 18, bail_y), (220, 220, 220), 2, cv2.LINE_AA)

    def _draw_lbw_projection(self, frame: np.ndarray, prediction: PredictionResult) -> None:
        path = np.array(prediction.predicted_points, dtype=np.int32)
        if len(path) >= 2:
            cv2.polylines(frame, [path], False, (0, 220, 255), 3, cv2.LINE_AA)
        cv2.line(frame, prediction.impact_point, prediction.collision_point, (0, 220, 255), 2, cv2.LINE_AA)
        label_x = min(prediction.impact_point[0] + 20, frame.shape[1] - 220)
        label_y = max(prediction.impact_point[1] - 26, 40)
        cv2.putText(frame, "PROJECTED BALL PATH", (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)

    def _draw_glow_dotted_line(
        self,
        frame: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        thickness: int,
        dot_spacing: int,
    ) -> None:
        if len(points) < 2:
            return

        for index in range(len(points) - 1):
            start = np.array(points[index], dtype=np.float32)
            end = np.array(points[index + 1], dtype=np.float32)
            distance = np.linalg.norm(end - start)
            if distance == 0:
                continue
            steps = max(int(distance // dot_spacing), 1)
            for step in range(steps + 1):
                ratio = step / steps
                point = start + (end - start) * ratio
                pixel = tuple(point.astype(int))
                cv2.circle(frame, pixel, thickness + 4, (40, 62, 118), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(frame, pixel, thickness + 1, color, thickness=-1, lineType=cv2.LINE_AA)
