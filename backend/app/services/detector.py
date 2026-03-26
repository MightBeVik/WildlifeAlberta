"""
Animal detector service using MegaDetector (YOLOv5) or mock fallback.
Stage 1 of the two-stage pipeline: Detects animals in camera trap images.
"""
import time
import random
import logging
from pathlib import Path
from typing import List, Optional

from app.config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD

logger = logging.getLogger(__name__)

# Try to import YOLO (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLO loaded successfully")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not available — using mock detector")


class WildlifeDetector:
    """
    Animal detection using YOLOv5/MegaDetector.
    Falls back to mock detections if model is not available.
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Attempt to load the YOLOv5 detection model."""
        if not YOLO_AVAILABLE:
            logger.info("Running in MOCK mode — no real detection model loaded")
            return

        try:
            # Use YOLOv5s as default — lightweight and fast
            # Replace with MegaDetector weights when available:
            # self.model = YOLO("path/to/megadetector_v5.pt")
            self.model = YOLO("yolov5su.pt")
            self.model_loaded = True
            logger.info("YOLOv5s detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            self.model_loaded = False

    def detect(self, image_path: str, confidence: float = None) -> dict:
        """
        Run detection on a single image.
        
        Returns:
            dict with keys: detections (list of bboxes), has_animal, processing_time_ms
        """
        conf = confidence or CONFIDENCE_THRESHOLD
        start = time.time()

        if self.model_loaded and self.model is not None:
            result = self._real_detect(image_path, conf)
        else:
            result = self._mock_detect(image_path)

        elapsed = (time.time() - start) * 1000
        result["processing_time_ms"] = round(elapsed, 2)
        return result

    def _real_detect(self, image_path: str, confidence: float) -> dict:
        """Run real YOLOv5 detection."""
        results = self.model(
            image_path,
            conf=confidence,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    xyxyn = box.xyxyn[0].tolist()  # normalized coords
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = r.names.get(cls, "unknown")

                    # Map YOLO/COCO classes to our categories
                    # MegaDetector uses: 0=animal, 1=person, 2=vehicle
                    # Standard YOLO/COCO has animal classes in various positions
                    category = self._map_category(cls, cls_name)

                    detections.append({
                        "x1": round(xyxyn[0], 4),
                        "y1": round(xyxyn[1], 4),
                        "x2": round(xyxyn[2], 4),
                        "y2": round(xyxyn[3], 4),
                        "confidence": round(conf, 4),
                        "category": category,
                    })

        # Filter to only animal detections for the main flag
        animal_detections = [d for d in detections if d["category"] == "animal"]

        return {
            "detections": detections,
            "has_animal": len(animal_detections) > 0,
        }

    def _map_category(self, cls_id: int, cls_name: str) -> str:
        """Map YOLO/COCO class to our categories: animal, person, vehicle."""
        # COCO person class
        if cls_id == 0 or cls_name == "person":
            return "person"

        # COCO vehicle classes
        vehicle_ids = {1, 2, 3, 4, 5, 6, 7, 8}  # bicycle, car, motorcycle, etc.
        vehicle_names = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
        if cls_id in vehicle_ids or cls_name in vehicle_names:
            return "vehicle"

        # COCO animal classes (14-23): bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
        animal_ids = set(range(14, 24))
        animal_names = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
        if cls_id in animal_ids or cls_name in animal_names:
            return "animal"

        # Default: treat as animal for camera trap context
        return "animal"

    def _mock_detect(self, image_path: str) -> dict:
        """
        Generate mock detections for demo/testing when model is not loaded.
        Simulates realistic camera trap detection results.
        """
        # ~70% chance of animal detected (simulating real camera trap data)
        has_animal = random.random() > 0.3

        detections = []
        if has_animal:
            num_animals = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
            for _ in range(num_animals):
                # Generate realistic bounding box (animal typically in center-ish area)
                cx = random.uniform(0.2, 0.8)
                cy = random.uniform(0.3, 0.8)
                w = random.uniform(0.1, 0.4)
                h = random.uniform(0.1, 0.4)

                detections.append({
                    "x1": round(max(0, cx - w / 2), 4),
                    "y1": round(max(0, cy - h / 2), 4),
                    "x2": round(min(1, cx + w / 2), 4),
                    "y2": round(min(1, cy + h / 2), 4),
                    "confidence": round(random.uniform(0.6, 0.98), 4),
                    "category": "animal",
                })

        return {
            "detections": detections,
            "has_animal": has_animal,
        }

    @property
    def is_loaded(self) -> bool:
        return self.model_loaded


# Singleton instance
detector_instance: Optional[WildlifeDetector] = None


def get_detector() -> WildlifeDetector:
    """Get or create the singleton detector instance."""
    global detector_instance
    if detector_instance is None:
        detector_instance = WildlifeDetector()
    return detector_instance
