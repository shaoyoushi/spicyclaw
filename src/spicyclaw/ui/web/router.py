"""Static file serving for the Web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter()


@router.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


def get_static_files() -> StaticFiles:
    return StaticFiles(directory=str(STATIC_DIR))
