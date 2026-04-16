from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.settings import settings

settings.output_videos_directory.mkdir(parents=True, exist_ok=True)
settings.temp_directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=settings.service_name, version=settings.version)
app.mount("/videos", StaticFiles(directory=settings.output_videos_directory), name="videos")
app.include_router(router)
