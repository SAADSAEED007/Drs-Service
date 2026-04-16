from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplayInstruction:
    frame_index: int
    stage: str
    zoom_scale: float = 1.0
    zoom_center: tuple[int, int] | None = None


class BroadcastReplayDirector:
    def build(self, frame_count: int, impact_frame_index: int, collision_point: tuple[int, int]) -> list[ReplayInstruction]:
        if frame_count <= 0:
            return []

        instructions: list[ReplayInstruction] = []
        intro_hold = min(10, max(frame_count // 12, 4))
        for _ in range(intro_hold):
            instructions.append(ReplayInstruction(frame_index=0, stage="decision_pending"))

        slow_motion_end = min(frame_count, max(impact_frame_index + 8, int(frame_count * 0.75)))
        for frame_index in range(slow_motion_end):
            instructions.append(ReplayInstruction(frame_index=frame_index, stage="tracking", zoom_scale=1.0))
            if frame_index >= max(impact_frame_index - 3, 0):
                instructions.append(ReplayInstruction(frame_index=frame_index, stage="tracking", zoom_scale=1.0))

        impact_zoom_start = max(impact_frame_index - 4, 0)
        impact_zoom_end = min(frame_count, impact_frame_index + 5)
        for frame_index in range(impact_zoom_start, impact_zoom_end):
            instructions.append(
                ReplayInstruction(
                    frame_index=frame_index,
                    stage="impact_zoom",
                    zoom_scale=1.45,
                    zoom_center=collision_point,
                )
            )

        for _ in range(max(12, frame_count // 10)):
            instructions.append(
                ReplayInstruction(
                    frame_index=frame_count - 1,
                    stage="projection",
                    zoom_scale=1.15,
                    zoom_center=collision_point,
                )
            )

        for _ in range(18):
            instructions.append(ReplayInstruction(frame_index=frame_count - 1, stage="decision"))

        return instructions
