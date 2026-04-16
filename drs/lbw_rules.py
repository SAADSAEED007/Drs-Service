from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from drs.calibrator import CalibrationResult, PitchCalibrator
from drs.constants import STUMP_WIDTH_M
from drs.impact_detector import ImpactResult
from drs.predictor import PredictionResult
from drs.tracker import TrackedPoint


@dataclass
class LBWVerdict:
    pitching: Literal["In Line", "Outside Leg", "Outside Off", "Unknown"]
    impact: Literal["In Line", "Outside Off", "Outside Leg", "Unknown"]
    wickets: Literal["Hitting", "Missing", "Umpire's Call", "Unknown"]
    decision: Literal["OUT", "NOT OUT", "UMPIRES CALL"]
    decision_reason: str
    confidence: float
    umpires_call: bool


class LBWRuleEngine:
    def __init__(self) -> None:
        self._calibrator = PitchCalibrator()

    def evaluate(
        self,
        calibration: CalibrationResult,
        trajectory: list[TrackedPoint],
        impact_result: ImpactResult,
        projection: PredictionResult,
    ) -> LBWVerdict:
        if not trajectory:
            return LBWVerdict(
                pitching="Unknown",
                impact="Unknown",
                wickets="Unknown",
                decision="NOT OUT",
                decision_reason="Ball trajectory was unavailable for LBW evaluation.",
                confidence=0.0,
                umpires_call=False,
            )

        pitch_point = trajectory[0]
        pitch_label = self._classify_line((pitch_point.x, pitch_point.y), calibration)
        impact_label = self._classify_line(impact_result.point_px, calibration)
        wicket_label, wicket_reason = self._classify_wicket_zone(projection)

        if pitch_label == "Outside Leg":
            return self._verdict(
                pitching=pitch_label,
                impact_label=impact_label,
                wickets=wicket_label,
                decision="NOT OUT",
                reason="Ball pitched outside leg stump.",
                calibration=calibration,
                impact_result=impact_result,
                projection=projection,
                umpires_call=False,
            )

        if impact_label == "Outside Off":
            return self._verdict(
                pitching=pitch_label,
                impact_label=impact_label,
                wickets=wicket_label,
                decision="NOT OUT",
                reason="Impact was outside off stump.",
                calibration=calibration,
                impact_result=impact_result,
                projection=projection,
                umpires_call=False,
            )

        if impact_label == "Outside Leg":
            return self._verdict(
                pitching=pitch_label,
                impact_label=impact_label,
                wickets=wicket_label,
                decision="NOT OUT",
                reason="Impact was outside leg stump with no shot-offered signal available.",
                calibration=calibration,
                impact_result=impact_result,
                projection=projection,
                umpires_call=False,
            )

        if wicket_label == "Hitting":
            return self._verdict(
                pitching=pitch_label,
                impact_label=impact_label,
                wickets=wicket_label,
                decision="OUT",
                reason=wicket_reason,
                calibration=calibration,
                impact_result=impact_result,
                projection=projection,
                umpires_call=False,
            )

        if wicket_label == "Umpire's Call":
            return self._verdict(
                pitching=pitch_label,
                impact_label=impact_label,
                wickets=wicket_label,
                decision="UMPIRES CALL",
                reason=wicket_reason,
                calibration=calibration,
                impact_result=impact_result,
                projection=projection,
                umpires_call=True,
            )

        return self._verdict(
            pitching=pitch_label,
            impact_label=impact_label,
            wickets=wicket_label,
            decision="NOT OUT",
            reason=wicket_reason,
            calibration=calibration,
            impact_result=impact_result,
            projection=projection,
            umpires_call=False,
        )

    def _classify_line(
        self,
        point_px: tuple[float, float],
        calibration: CalibrationResult,
    ) -> Literal["In Line", "Outside Leg", "Outside Off", "Unknown"]:
        try:
            line_x, _ = self._calibrator.to_pitch_space(point_px, calibration)
        except Exception:
            return "Unknown"

        half_stump = STUMP_WIDTH_M / 2.0
        if line_x < -half_stump:
            return "Outside Leg"
        if line_x > half_stump:
            return "Outside Off"
        return "In Line"

    def _classify_wicket_zone(
        self,
        projection: PredictionResult,
    ) -> tuple[Literal["Hitting", "Missing", "Umpire's Call", "Unknown"], str]:
        overlap_ratio = float(max(0.0, min(projection.wicket_overlap_ratio, 1.0)))
        if overlap_ratio > 0.5:
            return "Hitting", "Projected ball path overlaps the wicket zone by more than 50%."
        if overlap_ratio > 0.0:
            return "Umpire's Call", "Projected ball path clips the wicket zone inside the umpire's-call margin."
        return "Missing", "Projected ball path does not overlap the wicket zone."

    def _verdict(
        self,
        pitching: Literal["In Line", "Outside Leg", "Outside Off", "Unknown"],
        impact_label: Literal["In Line", "Outside Off", "Outside Leg", "Unknown"],
        wickets: Literal["Hitting", "Missing", "Umpire's Call", "Unknown"],
        decision: Literal["OUT", "NOT OUT", "UMPIRES CALL"],
        reason: str,
        calibration: CalibrationResult,
        impact_result: ImpactResult,
        projection: PredictionResult,
        umpires_call: bool,
    ) -> LBWVerdict:
        decision_confidence = float(
            max(
                0.0,
                min(
                    calibration.confidence
                    * max(impact_result.confidence, 0.1)
                    * max(projection.projection_confidence, 0.1),
                    0.99,
                ),
            )
        )
        return LBWVerdict(
            pitching=pitching,
            impact=impact_label,
            wickets=wickets,
            decision=decision,
            decision_reason=reason,
            confidence=round(decision_confidence, 2),
            umpires_call=umpires_call,
        )
