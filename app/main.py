"""FastAPI application entrypoint."""

import logging
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

_STATIC = Path(__file__).parent / "static"
_ROOT = Path(__file__).parent.parent

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup checks then yield control to the application.

    Logs warnings for missing config or unreachable Statewave but does not block
    startup, so developers get a clear message rather than a cryptic first-request crash.
    """
    if not settings.llm_configured:
        logger.warning(
            "Startup: LLM_API_KEY is not set. "
            "Copy .env.example to .env and fill in your API key."
        )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.statewave_base_url}/healthz")
            if resp.status_code >= 400:
                logger.warning(
                    "Startup: Statewave at %s returned HTTP %d. "
                    "Is it running? See Quick Start step 3 in the README.",
                    settings.statewave_base_url,
                    resp.status_code,
                )
            else:
                logger.info(
                    "Startup: Statewave reachable at %s. model=%s",
                    settings.statewave_base_url,
                    settings.openai_model,
                )
    except httpx.ConnectError:
        logger.warning(
            "Startup: Cannot connect to Statewave at %s. "
            "Is Docker running? See Quick Start step 3 in the README.",
            settings.statewave_base_url,
        )
    except httpx.TimeoutException:
        logger.warning(
            "Startup: Statewave at %s did not respond within 5 seconds.",
            settings.statewave_base_url,
        )

    yield  # application runs here


app = FastAPI(
    title="Statewave Personal Assistant",
    description=(
        "Personal assistant with persistent cross-session memory powered by Statewave + FastAPI. "
        "Demonstrates ranked, token-bounded memory retrieval as a drop-in replacement for "
        "naive chat-history injection."
    ),
    version="0.1.0",
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# CORS: restricted to configured origins (defaults to localhost only).
# Set CORS_ORIGINS in .env for production (e.g. "https://yourapp.com").
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach a unique request ID to every request and propagate it in the response header.

    If the client sends X-Request-ID, that value is used (allows end-to-end tracing).
    Otherwise a new UUID is generated. Appears in all log lines and response headers
    so it can be correlated across client and server logs.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    logger.debug(
        "request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/demo", include_in_schema=False)
async def demo() -> FileResponse:
    return FileResponse(_STATIC / "demo.html")


@app.get("/integrate", include_in_schema=False)
async def integrate() -> FileResponse:
    return FileResponse(_STATIC / "integrate.html")


@app.get("/static/statewave_integration.py", include_in_schema=False)
async def download_integration() -> FileResponse:
    return FileResponse(
        _ROOT / "statewave_integration.py",
        media_type="text/x-python",
        filename="statewave_integration.py",
    )
