from pathlib import Path
import shutil
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.settings import settings
from app.models.schemas import DrsAnalysisResponse, HealthResponse
from drs.processor import DrsProcessor

router = APIRouter()
processor = DrsProcessor(settings.output_videos_directory)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name, version=settings.version)


@router.post("/drs/analyze", response_model=DrsAnalysisResponse)
async def analyze_drs(video: UploadFile = File(...)) -> DrsAnalysisResponse:
    suffix = Path(video.filename or "review.mp4").suffix or ".mp4"
    settings.temp_directory.mkdir(parents=True, exist_ok=True)
    temp_path = settings.temp_directory / f"drs-upload-{uuid4().hex}{suffix}"

    try:
        with temp_path.open("wb") as temp_file:
            shutil.copyfileobj(video.file, temp_file)
        result = processor.process(temp_path)
        return DrsAnalysisResponse(
            decision=result.decision,
            video_url=result.video_url,
            confidence=result.confidence,
            tracking_confidence=result.tracking_confidence,
            decision_confidence=result.decision_confidence,
            trajectory=[{"x": point[0], "y": point[1]} for point in result.trajectory],
            predicted_trajectory=[{"x": point[0], "y": point[1]} for point in result.predicted_trajectory],
            raw_trajectory=result.raw_trajectory,
            projection_points=[{"x": point[0], "y": point[1]} for point in result.projection_points],
            impact_point={"x": result.impact_point[0], "y": result.impact_point[1]},
            pitching=result.pitching,
            impact=result.impact,
            wickets=result.wickets,
            ultra_edge=result.ultra_edge,
            decision_reason=result.decision_reason,
            umpires_call=result.umpires_call,
            calibration_method=result.calibration_method,
            pitching_raw=result.pitching,
            impact_raw=result.impact,
            wickets_raw=result.wickets,
            metadata={
                "tracked_frames": len(result.trajectory),
                "total_frames_sampled": result.frame_count,
                "fps": result.fps,
                "impact_index": result.impact_frame_index,
                "projected_wicket_point": {
                    "x": result.predicted_trajectory[-1][0] if result.predicted_trajectory else result.impact_point[0],
                    "y": result.predicted_trajectory[-1][1] if result.predicted_trajectory else result.impact_point[1],
                },
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Failed to analyze DRS video") from error
    finally:
        await video.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
