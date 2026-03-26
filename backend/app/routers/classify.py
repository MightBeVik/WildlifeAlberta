"""
Classification router — runs species classification on detected animal crops.
"""
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.services.classifier import get_classifier
from app.services.preprocessor import load_image, crop_detection
from app.routers.upload import get_uploaded_images
from app.routers.detect import get_detection_results

router = APIRouter(prefix="/api", tags=["classification"])

# In-memory classification results store
classification_results: dict = {}


@router.post("/classify/{image_id}")
async def classify_species(image_id: str):
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

    detection = detections[image_id]
    if not detection["has_animal"]:
        return {
            "image_id": image_id,
            "message": "No animals detected in this image",
            "classifications": [],
        }

    # Load the original image
    filepath = images[image_id]["filepath"]
    img = load_image(Path(filepath))

    # Classify each animal detection
    classifier = get_classifier()
    classifications = []

    animal_detections = [d for d in detection["detections"] if d["category"] == "animal"]

    for i, det in enumerate(animal_detections):
        # Crop the detection region
        bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
        crop = crop_detection(img, bbox, padding=0.1)

        # Run classification
        result = classifier.classify(crop)
        result["detection_index"] = i
        result["bbox"] = det
        classifications.append(result)

    # Store results
    classification_data = {
        "image_id": image_id,
        "filename": images[image_id]["filename"],
        "image_url": f"/uploads/{images[image_id]['saved_as']}",
        "classifications": classifications,
        "total_animals": len(classifications),
        "timestamp": datetime.now().isoformat(),
        "detections": detection["detections"],
        "width": images[image_id]["width"],
        "height": images[image_id]["height"],
    }
    classification_results[image_id] = classification_data

    return classification_data


@router.get("/classifications")
async def list_classifications():
    """List all classification results."""
    return {
        "classifications": list(classification_results.values()),
        "total": len(classification_results),
    }


@router.get("/classifications/{image_id}")
async def get_classification(image_id: str):
    """Get classification result for a specific image."""
    if image_id not in classification_results:
        raise HTTPException(status_code=404, detail="No classification results for this image")
    return classification_results[image_id]


@router.get("/stats")
async def get_stats():
    """Get overall system statistics."""
    images = get_uploaded_images()
    detections = get_detection_results()

    # Count species
    species_counts = {}
    for result in classification_results.values():
        for cls in result["classifications"]:
            species = cls["species"]
            species_counts[species] = species_counts.get(species, 0) + 1

    # Count images with animals vs empty
    images_with_animals = sum(1 for d in detections.values() if d["has_animal"])
    empty_images = sum(1 for d in detections.values() if not d["has_animal"])

    return {
        "total_images": len(images),
        "total_processed": len(detections),
        "images_with_animals": images_with_animals,
        "empty_images": empty_images,
        "total_classifications": len(classification_results),
        "species_counts": species_counts,
        "recent_uploads": list(images.values())[-5:],
        "model_info": {
            "detector": "YOLOv5 (MegaDetector)" if get_classifier().is_loaded else "Mock (demo mode)",
            "classifier": "EfficientNet-B3" if get_classifier().is_loaded else "Mock (demo mode)",
        },
    }
