from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.backend import routes
from app.backend import websockets
from app.backend.analysis import BrowserAnalysisService
from app.backend.errors import register_error_handlers
from app.backend.middleware import RequestIdMiddleware
from app.backend.middleware import SecurityHeadersMiddleware
from app.backend.services import StreamService
from app.core.config import AppConfig


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend"


def create_app(
    config: AppConfig | None = None,
    stream_service: StreamService | None = None,
    analysis_service: BrowserAnalysisService | None = None,
) -> FastAPI:
    """Application factory: collaborators are created here (composition root) or injected by callers/tests."""
    config = config or AppConfig()
    stream_service = stream_service or StreamService(config)
    analysis_service = analysis_service or BrowserAnalysisService(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not config.auth_enabled:
            LOGGER.warning(
                "API tokens are not configured: every endpoint is open. "
                "Set EMOTION_API_READ_TOKEN and EMOTION_API_ADMIN_TOKEN before exposing this server."
            )
        yield
        await stream_service.shutdown()

    app = FastAPI(
        title="EmotionAI API",
        version=__version__,
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.stream_service = stream_service
    app.state.analysis_service = analysis_service

    # Last added = outermost: request IDs wrap everything, including error responses.
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        # Auth uses headers, not cookies: credentials would only widen CORS exposure (and are unsafe with "*").
        allow_credentials=False,
    )
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app)

    app.include_router(routes.router)
    app.include_router(websockets.router)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

    @app.get("/", include_in_schema=False)
    async def get_index() -> FileResponse:
        return FileResponse(FRONTEND_PATH / "index.html")

    return app


app = create_app()
