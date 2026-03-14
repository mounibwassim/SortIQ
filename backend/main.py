import os
from fastapi import FastAPI, Depends, Request  # pyre-ignore
from fastapi.middleware.cors import CORSMiddleware  # pyre-ignore
from fastapi.staticfiles import StaticFiles  # pyre-ignore
from contextlib import asynccontextmanager

from database import create_tables  # pyre-ignore
from model_loader import get_model  # pyre-ignore
from logger import logger  # pyre-ignore
from routers import predict, stats, history, health, settings  # pyre-ignore

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up SortIQ Backend...")
    # Ensure upload directories exist
    os.makedirs("uploads/thumbnails", exist_ok=True)
    # Initialize DB tables
    create_tables()
    
    # Load ML Model (which includes warm-up)
    model = get_model()
    model.load()
    
    yield
    logger.info("Shutting down SortIQ Backend...")

app = FastAPI(
    title="SortIQ API",
    description="Smart Recycling Waste Classification API",
    version=os.getenv("MODEL_VERSION", "v1.0"),
    lifespan=lifespan
)

# CORS Setup — allow all origins for local dev, or specific ones in production
cors_env = os.getenv("CORS_ORIGINS", "")
if cors_env:
    origins = [o.strip() for o in cors_env.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Default: allow all origins so any Vite dev port (5173, 5174, etc.) works
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "https://sortiq-web.vercel.app",
            "https://*.vercel.app",
            "*"
        ],
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

@app.get("/")
def read_root():
    return {"message": "Welcome to SortIQ API"}
