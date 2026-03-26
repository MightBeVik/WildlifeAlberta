"""
FastAPI main application — Wildlife Detection & Classification System
eDNA Research Lab, SAIT — Camera Trap Analysis Platform
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, UPLOAD_DIR
from app.routers import upload, detect, classify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Wildlife Detection & Classification API",
    description=(
        "Camera trap image analysis for the eDNA Research Lab at SAIT. "
        "Detects and classifies wildlife species from motion-activated camera images "
        "across Alberta's forests and rural farmlands."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images as static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Include routers
app.include_router(upload.router)
app.include_router(detect.router)
app.include_router(classify.router)


@app.get("/")
async def root():
    return {
        "name": "Wildlife Detection & Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /api/upload",
            "detect": "POST /api/detect/{image_id}",
            "classify": "POST /api/classify/{image_id}",
            "images": "GET /api/images",
            "detections": "GET /api/detections",
            "classifications": "GET /api/classifications",
            "stats": "GET /api/stats",
        },
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint with model status."""
    try:
        from app.services.detector import get_detector
        from app.services.classifier import get_classifier

        detector = get_detector()
        classifier = get_classifier()

        return {
            "status": "healthy",
            "detector_loaded": detector.is_loaded,
            "classifier_loaded": classifier.is_loaded,
            "version": "1.0.0",
            "mode": "production" if (detector.is_loaded and classifier.is_loaded) else "demo",
        }
    except Exception:
        return {
            "status": "healthy",
            "detector_loaded": False,
            "classifier_loaded": False,
            "version": "1.0.0",
            "mode": "demo",
        }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Wildlife Detection & Classification System")
    logger.info("eDNA Research Lab — SAIT")
    logger.info("=" * 60)
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info("Loading models...")

    # Pre-load models on startup
    from app.services.detector import get_detector
    from app.services.classifier import get_classifier

    detector = get_detector()
    classifier = get_classifier()

    if detector.is_loaded:
        logger.info("✓ Detector model loaded (YOLOv5)")
    else:
        logger.info("⚠ Detector running in MOCK mode")

    if classifier.is_loaded:
        logger.info("✓ Classifier model loaded (EfficientNet-B3)")
    else:
        logger.info("⚠ Classifier running in MOCK mode")

    logger.info("=" * 60)
    logger.info("API ready at http://localhost:8000")
    logger.info("API docs at http://localhost:8000/docs")
    logger.info("=" * 60)
