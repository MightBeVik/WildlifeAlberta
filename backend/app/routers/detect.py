"""Detection router — runs one or more detector models on uploaded images."""
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.config import CONFIDENCE_THRESHOLD
from app.services.detector import get_detector
from app.routers.upload import get_uploaded_images

router = APIRouter(prefix="/api", tags=["detection"])

# In-memory detection results store
detection_results: dict = {}


@router.post("/detect/{image_id}")
async def detect_animals(
    image_id: str,
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.1, le=1.0),
    models: str = Query(default="all"),
):
    """Run animal detection on a specific uploaded image."""
    images = get_uploaded_images()
    if image_id not in images:
        raise HTTPException(status_code=404, detail="Image not found. Upload an image first.")

    image_info = images[image_id]
    filepath = image_info["filepath"]

    if not Path(filepath).exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    detector = get_detector()
    model_bundle = detector.detect_all(filepath, confidence=confidence, detector_keys=models)
    primary_key = model_bundle["primary_detector"]
    primary_result = model_bundle["by_detector"][primary_key]

    detection_data = {
        "image_id": image_id,
        "filename": image_info["filename"],
        "image_url": f"/uploads/{image_info['saved_as']}",
        "detections": primary_result["detections"],
        "has_animal": primary_result["has_animal"],
        "processing_time_ms": primary_result["processing_time_ms"],
        "detector_key": primary_result["detector_key"],
        "detector_label": primary_result["detector_label"],
        "detector_mode": primary_result["detector_mode"],
        "primary_detector": primary_key,
        "detector_order": model_bundle["detector_order"],
        "available_detectors": model_bundle["available_detectors"],
        "by_detector": model_bundle["by_detector"],
        "comparisons": list(model_bundle["by_detector"].values()),
        "timestamp": datetime.now().isoformat(),
        "width": image_info["width"],
        "height": image_info["height"],
    }
    detection_results[image_id] = detection_data

    return detection_data


@router.post("/detect/batch")
async def detect_batch(
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.1, le=1.0),
    models: str = Query(default="all"),
):
    """Run detection on all uploaded images that haven't been processed yet."""
    images = get_uploaded_images()
    results = []

    for image_id in images:
        if image_id not in detection_results:
            try:
                result = await detect_animals(image_id, confidence, models)
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
