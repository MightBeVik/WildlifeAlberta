"""Classification router — runs species classification per detector result."""
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.services.classifier import get_classifier
from app.services.preprocessor import load_image, crop_detection
from app.routers.upload import get_uploaded_images
from app.routers.detect import get_detection_results

router = APIRouter(prefix="/api", tags=["classification"])

# In-memory classification results store
classification_results: dict = {}


@router.post("/classify/{image_id}")
async def classify_species(image_id: str, detector_key: str = Query(default="primary")):
    """
    Run species classification on detected animals in an image.
    Requires detection to have been run first.
    """
    detections = get_detection_results()
    images = get_uploaded_images()

    if image_id not in images:
        raise HTTPException(status_code=404, detail="Image not found")

    if image_id not in detections:
        raise HTTPException(
            status_code=400,
            detail="No detections found. Run detection first: POST /api/detect/{image_id}",
        )

    detection_bundle = detections[image_id]
    selected_key = detection_bundle["primary_detector"] if detector_key == "primary" else detector_key
    if selected_key == "all":
        selected_keys = detection_bundle["detector_order"]
    else:
        selected_keys = [selected_key]

    filepath = images[image_id]["filepath"]
    img = load_image(Path(filepath))
    classifier = get_classifier()

    image_classifications = classification_results.setdefault(image_id, {})
    by_detector = {}

    for key in selected_keys:
        detection = detection_bundle["by_detector"].get(key)
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Detection results not found for detector '{key}'")

        if not detection["has_animal"]:
            classification_data = {
                "image_id": image_id,
                "filename": images[image_id]["filename"],
                "image_url": f"/uploads/{images[image_id]['saved_as']}",
                "detector_key": key,
                "detector_label": detection["detector_label"],
                "detector_mode": detection["detector_mode"],
                "classifications": [],
                "total_animals": 0,
                "timestamp": datetime.now().isoformat(),
                "detections": detection["detections"],
                "width": images[image_id]["width"],
                "height": images[image_id]["height"],
                "has_animal": False,
                "processing_time_ms": detection["processing_time_ms"],
                "message": "No animals detected for this detector",
            }
            image_classifications[key] = classification_data
            by_detector[key] = classification_data
            continue

        classifications = []
        animal_detections = [d for d in detection["detections"] if d["category"] == "animal"]

        for index, det in enumerate(animal_detections):
            bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
            crop = crop_detection(img, bbox, padding=0.1)
            result = classifier.classify(crop)
            result["detection_index"] = index
            result["bbox"] = det
            classifications.append(result)

        classification_data = {
            "image_id": image_id,
            "filename": images[image_id]["filename"],
            "image_url": f"/uploads/{images[image_id]['saved_as']}",
            "detector_key": key,
            "detector_label": detection["detector_label"],
            "detector_mode": detection["detector_mode"],
            "classifications": classifications,
            "total_animals": len(classifications),
            "timestamp": datetime.now().isoformat(),
            "detections": detection["detections"],
            "width": images[image_id]["width"],
            "height": images[image_id]["height"],
            "has_animal": True,
            "processing_time_ms": detection["processing_time_ms"],
        }
        image_classifications[key] = classification_data
        by_detector[key] = classification_data

    if len(selected_keys) == 1:
        return by_detector[selected_keys[0]]

    primary_key = detection_bundle["primary_detector"]
    primary_result = by_detector.get(primary_key) or by_detector[selected_keys[0]]
    return {
        **primary_result,
        "primary_detector": primary_key,
        "detector_order": list(by_detector.keys()),
        "by_detector": by_detector,
    }


@router.get("/classifications")
async def list_classifications():
    """List all classification results."""
    flattened = []
    for per_image in classification_results.values():
        flattened.extend(per_image.values())

    return {
        "classifications": flattened,
        "total": len(flattened),
    }


@router.get("/classifications/{image_id}")
async def get_classification(image_id: str, detector_key: str = Query(default="primary")):
    """Get classification result for a specific image."""
    if image_id not in classification_results:
        raise HTTPException(status_code=404, detail="No classification results for this image")

    if detector_key == "primary":
        detections = get_detection_results()
        primary_key = detections.get(image_id, {}).get("primary_detector")
        if primary_key and primary_key in classification_results[image_id]:
            return classification_results[image_id][primary_key]

    if detector_key == "all":
        return {
            "image_id": image_id,
            "by_detector": classification_results[image_id],
        }

    if detector_key not in classification_results[image_id]:
        raise HTTPException(status_code=404, detail=f"No classification results for detector '{detector_key}'")

    return classification_results[image_id][detector_key]


@router.get("/stats")
async def get_stats():
    """Get overall system statistics."""
    images = get_uploaded_images()
    detections = get_detection_results()
    detector_manager = get_detector()

    # Count species
    species_counts = {}
    for image_id, result_map in classification_results.items():
        primary_key = detections.get(image_id, {}).get("primary_detector")
        result = result_map.get(primary_key) if primary_key else next(iter(result_map.values()), None)
        if result is None:
            continue
        for cls in result["classifications"]:
            species = cls["species"]
            species_counts[species] = species_counts.get(species, 0) + 1

    # Count images with animals vs empty
    images_with_animals = sum(1 for d in detections.values() if d.get("has_animal"))
    empty_images = sum(1 for d in detections.values() if not d.get("has_animal"))

    return {
        "total_images": len(images),
        "total_processed": len(detections),
        "images_with_animals": images_with_animals,
        "empty_images": empty_images,
        "total_classifications": sum(len(per_image) for per_image in classification_results.values()),
        "species_counts": species_counts,
        "recent_uploads": list(images.values())[-5:],
        "model_info": {
            "detector": detector_manager.list_available_detectors(),
            "classifier": "EfficientNet-B3" if get_classifier().is_loaded else "Mock (demo mode)",
        },
    }
