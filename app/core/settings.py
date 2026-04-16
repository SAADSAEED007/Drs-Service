import os
from pathlib import Path


class Settings:
    service_name = "khelaao-drs-service"
    version = "1.0.0"
    temp_directory = Path("tmp")
    output_directory = Path("public")
    output_videos_directory = output_directory / "videos"
    models_directory = Path(os.getenv("DRS_MODELS_DIRECTORY", "models"))
    yolo_model_path = os.getenv("DRS_YOLO_MODEL")
    enable_stabilization = os.getenv("DRS_ENABLE_STABILIZATION", "1") != "0"
    enable_replay_audio = os.getenv("DRS_ENABLE_REPLAY_AUDIO", "1") != "0"


settings = Settings()
