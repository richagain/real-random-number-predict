"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.config import APP_TITLE, APP_DESCRIPTION, BASE_DIR, DEBUG, SCHEDULER_ENABLED
from app.database.db import init_db
from app.api.routes import router as api_router
from app.scheduler import setup_scheduler, shutdown_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    if SCHEDULER_ENABLED:
        setup_scheduler()
    yield
    # Shutdown
    if SCHEDULER_ENABLED:
        shutdown_scheduler()


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    debug=DEBUG,
)

# Mount static files
static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Templates
templates_path = BASE_DIR / "app" / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    """Render history page."""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/statistics", response_class=HTMLResponse)
async def statistics_page(request: Request):
    """Render statistics page."""
    return templates.TemplateResponse("statistics.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Render prediction page."""
    return templates.TemplateResponse("predict.html", {"request": request})
