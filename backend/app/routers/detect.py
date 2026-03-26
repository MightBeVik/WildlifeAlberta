"""
Detection router — runs MegaDetector on uploaded images.
"""
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.config import UPLOAD_DIR, CONFIDENCE_THRESHOLD
from app.services.detector import get_detector
from app.services.preprocessor import load_image
from app.models.schemas import Detection
from app.routers.upload import get_uploaded_images

router = APIRouter(prefix="/api", tags=["detection"])

# In-memory detection results store
detection_results: dict = {}


@router.post("/detect/{image_id}")
async def detect_animals(
    image_id: str,
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.1, le=1.0),
):
    """Run animal detection on a specific uploaded image."""
    images = get_uploaded_images()
    if image_id not in images:
        raise HTTPException(status_code=404, detail="Image not found. Upload an image first.")

    image_info = images[image_id]
    filepath = image_info["filepath"]

    if not Path(filepath).exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Run detection
    detector = get_detector()
    result = detector.detect(filepath, confidence=confidence)

    # Store result
    detection_data = {
        "image_id": image_id,
        "filename": image_info["filename"],
        "image_url": f"/uploads/{image_info['saved_as']}",
        "detections": result["detections"],
        "has_animal": result["has_animal"],
        "processing_time_ms": result["processing_time_ms"],
        "timestamp": datetime.now().isoformat(),
        "width": image_info["width"],
        "height": image_info["height"],
    }
    detection_results[image_id] = detection_data

    return detection_data


@router.post("/detect/batch")
async def detect_batch(
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.1, le=1.0),
):
    """Run detection on all uploaded images that haven't been processed yet."""
    images = get_uploaded_images()
    results = []

    for image_id in images:
        if image_id not in detection_results:
            try:
                result = await detect_animals(image_id, confidence)
                results.append(result)
            except Exception as e:
                results.append({"image_id": image_id, "error": str(e)})

    return {
        "results": results,
        "processed": len(results),
        "total_images": len(images),
    }


@router.get("/detections")
async def list_detections():
    """List all detection results."""
    return {
        "detections": list(detection_results.values()),
        "total": len(detection_results),
    }


@router.get("/detections/{image_id}")
async def get_detection(image_id: str):
    """Get detection result for a specific image."""
    if image_id not in detection_results:
        raise HTTPException(status_code=404, detail="No detection results for this image")
    return detection_results[image_id]


def get_detection_results():
    """Access detection results from other modules."""
    return detection_results
