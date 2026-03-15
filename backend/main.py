import os
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from database import create_tables
from model_loader import get_model
from logger import logger
from routers import predict, stats, history, health, settings

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SortIQ starting up - models will load on first request")
    # Ensure upload directories exist
    os.makedirs("uploads/thumbnails", exist_ok=True)
    # Initialize DB tables
    create_tables()
    
    yield
    logger.info("Shutting down SortIQ Backend...")

app = FastAPI(
    title="SortIQ API",
    description="Smart Recycling Waste Classification API",
    version=os.getenv("MODEL_VERSION", "v1.0"),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(stats.router, prefix="/stats", tags=["Analytics"])
app.include_router(history.router, prefix="/history", tags=["History"])
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(settings.router, prefix="/settings", tags=["Settings"])

# Serve thumbnail images from disk
os.makedirs("uploads/thumbnails", exist_ok=True)
app.mount("/static/thumbnails", StaticFiles(directory="uploads/thumbnails"), name="thumbnails")

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint to verify API is active.
    """
    return {
        "status": "online",
        "message": "SortIQ AI Backend is running",
        "docs": "/docs",
        "health": "/health"
    }