"""Detector service with support for multiple camera-trap comparison models."""
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.config import (
    CAMERA_TRAP_DETECTOR_WEIGHTS,
    CONFIDENCE_THRESHOLD,
    GENERIC_YOLO_WEIGHTS,
    IOU_THRESHOLD,
    MEGADETECTOR_WEIGHTS,
)

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLO loaded successfully")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not available — using mock detector")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    label: str
    weights: str
    description: str
    priority: int
    requires_local_file: bool = False


class WildlifeDetector:
    """Loads one concrete YOLO-compatible detector."""

    def __init__(self, spec: DetectorSpec):
        self.spec = spec
        self.model = None
        self.model_backend = "unloaded"
        self.model_loaded = False
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_model(self):
        weights_path = Path(self.spec.weights)
        if self.spec.requires_local_file and not weights_path.exists():
            self.load_error = f"weights not found at {weights_path}"
            logger.info("%s unavailable: %s", self.spec.label, self.load_error)
            return

        try:
            if self.spec.key == "megadetector":
                if not TORCH_AVAILABLE:
                    self.load_error = "torch is not installed in the active backend environment"
                    return

                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=self.spec.weights,
                    trust_repo=True,
                )
                self.model_backend = "legacy_yolov5"
            else:
                if not YOLO_AVAILABLE:
                    self.load_error = "ultralytics is not installed in the active backend environment"
                    return

                self.model = YOLO(self.spec.weights)
                self.model_backend = "ultralytics"

            self.model_loaded = True
            logger.info("Loaded detector '%s' from %s", self.spec.label, self.spec.weights)
        except Exception as exc:
            self.load_error = str(exc)
            self.model_loaded = False
            logger.error("Failed to load detector '%s': %s", self.spec.label, exc)

    def detect(self, image_path: str, confidence: Optional[float] = None) -> dict:
        conf = confidence or CONFIDENCE_THRESHOLD
        start = time.time()
        if self.model_backend == "legacy_yolov5":
            result = self._legacy_detect(image_path, conf)
        else:
            result = self._real_detect(image_path, conf)

        elapsed = (time.time() - start) * 1000
        result.update(
            {
                "detector_key": self.spec.key,
                "detector_label": self.spec.label,
                "detector_description": self.spec.description,
                "detector_mode": "real",
                "detector_backend": self.model_backend,
                "weights": self.spec.weights,
                "processing_time_ms": round(elapsed, 2),
            }
        )
        return result

    def _real_detect(self, image_path: str, confidence: float) -> dict:
        results = self.model(
            image_path,
            conf=confidence,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxyn = box.xyxyn[0].tolist()
                box_conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = result.names.get(cls, "unknown")
                category = self._map_category(cls, cls_name)
                detections.append(
                    {
                        "x1": round(xyxyn[0], 4),
                        "y1": round(xyxyn[1], 4),
                        "x2": round(xyxyn[2], 4),
                        "y2": round(xyxyn[3], 4),
                        "confidence": round(box_conf, 4),
                        "category": category,
                    }
                )

        animal_detections = [d for d in detections if d["category"] == "animal"]
        return {"detections": detections, "has_animal": len(animal_detections) > 0}

    def _legacy_detect(self, image_path: str, confidence: float) -> dict:
        results = self.model(image_path, size=1280)
        raw_detections = getattr(results, "xyxyn", [])[0].tolist()
        names = getattr(results, "names", {})

        detections = []
        for x1, y1, x2, y2, box_conf, cls in raw_detections:
            cls_id = int(cls)
            if isinstance(names, dict):
                cls_name = names.get(cls_id, "unknown")
            else:
                cls_name = names[cls_id] if cls_id < len(names) else "unknown"

            if float(box_conf) < confidence:
                continue

            detections.append(
                {
                    "x1": round(float(x1), 4),
                    "y1": round(float(y1), 4),
                    "x2": round(float(x2), 4),
                    "y2": round(float(y2), 4),
                    "confidence": round(float(box_conf), 4),
                    "category": self._map_category(cls_id, cls_name),
                }
            )

        animal_detections = [d for d in detections if d["category"] == "animal"]
        return {"detections": detections, "has_animal": len(animal_detections) > 0}

    def _map_category(self, cls_id: int, cls_name: str) -> str:
        if self.spec.key == "megadetector":
            if cls_id == 0 or cls_name == "animal":
                return "animal"
            if cls_id == 1 or cls_name == "person":
                return "person"
            if cls_id == 2 or cls_name == "vehicle":
                return "vehicle"

        if cls_id == 0 or cls_name == "person":
            return "person"

        vehicle_ids = {1, 2, 3, 4, 5, 6, 7, 8}
        vehicle_names = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
        if cls_id in vehicle_ids or cls_name in vehicle_names:
            return "vehicle"

        animal_ids = set(range(14, 24))
        animal_names = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
        if cls_id in animal_ids or cls_name in animal_names:
            return "animal"

        return "animal"

    @property
    def is_loaded(self) -> bool:
        return self.model_loaded

    def summary(self) -> dict:
        return {
            "key": self.spec.key,
            "label": self.spec.label,
            "description": self.spec.description,
            "weights": self.spec.weights,
            "is_loaded": self.model_loaded,
            "mode": "real" if self.model_loaded else "unavailable",
            "backend": self.model_backend,
            "error": self.load_error,
        }


class MultiDetectorManager:
    """Runs one image through all available real detectors, with mock fallback only if needed."""

    def __init__(self):
        self.detectors: Dict[str, WildlifeDetector] = {}
        for spec in self._build_specs():
            self.detectors[spec.key] = WildlifeDetector(spec)

    def _build_specs(self) -> List[DetectorSpec]:
        specs = [
            DetectorSpec(
                key="camera_trap_yolo",
                label="Camera Trap YOLO",
                weights=CAMERA_TRAP_DETECTOR_WEIGHTS,
                description="Preferred slot for a YOLO model fine-tuned on day/night wildlife camera-trap imagery.",
                priority=0,
                requires_local_file=True,
            ),
            DetectorSpec(
                key="megadetector",
                label="MegaDetector",
                weights=MEGADETECTOR_WEIGHTS,
                description="Camera-trap detector specialized for animals, people, and vehicles.",
                priority=1,
                requires_local_file=True,
            ),
            DetectorSpec(
                key="generic_yolo",
                label="Generic YOLO",
                weights=GENERIC_YOLO_WEIGHTS,
                description="General-purpose pretrained YOLO baseline for comparison.",
                priority=2,
                requires_local_file=False,
            ),
        ]
        return sorted(specs, key=lambda spec: spec.priority)

    def list_available_detectors(self) -> List[dict]:
        detectors = [detector.summary() for detector in self.detectors.values()]
        detectors.append(
            {
                "key": "mock_detector",
                "label": "Mock Detector",
                "description": "Fallback demo detector used only when no real detector loads.",
                "weights": None,
                "is_loaded": not self.is_loaded,
                "mode": "mock",
                "error": None if not self.is_loaded else "disabled while real detectors are available",
            }
        )
        return detectors

    def resolve_detector_keys(self, detector_keys: Optional[str] = None) -> List[str]:
        loaded_keys = [key for key, detector in self.detectors.items() if detector.is_loaded]

        if detector_keys is None or detector_keys == "all":
            return loaded_keys or ["mock_detector"]

        requested = [key.strip() for key in detector_keys.split(",") if key.strip()]
        selected = [key for key in requested if key == "mock_detector" or key in self.detectors]

        if not selected:
            return loaded_keys or ["mock_detector"]

        available = [key for key in selected if key == "mock_detector" or self.detectors[key].is_loaded]
        return available or ["mock_detector"]

    def detect_all(self, image_path: str, confidence: Optional[float] = None, detector_keys: Optional[str] = None) -> dict:
        selected_keys = self.resolve_detector_keys(detector_keys)
        by_detector = {}

        for key in selected_keys:
            if key == "mock_detector":
                by_detector[key] = self._mock_detect(image_path)
                continue

            by_detector[key] = self.detectors[key].detect(image_path, confidence=confidence)

        primary_detector = selected_keys[0]
        return {
            "primary_detector": primary_detector,
            "detector_order": selected_keys,
            "by_detector": by_detector,
            "available_detectors": self.list_available_detectors(),
        }

    def _mock_detect(self, image_path: str) -> dict:
        has_animal = random.random() > 0.3
        detections = []

        if has_animal:
            num_animals = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
            for _ in range(num_animals):
                cx = random.uniform(0.2, 0.8)
                cy = random.uniform(0.3, 0.8)
                width = random.uniform(0.1, 0.4)
                height = random.uniform(0.1, 0.4)
                detections.append(
                    {
                        "x1": round(max(0, cx - width / 2), 4),
                        "y1": round(max(0, cy - height / 2), 4),
                        "x2": round(min(1, cx + width / 2), 4),
                        "y2": round(min(1, cy + height / 2), 4),
                        "confidence": round(random.uniform(0.6, 0.98), 4),
                        "category": "animal",
                    }
                )

        return {
            "detector_key": "mock_detector",
            "detector_label": "Mock Detector",
            "detector_description": "Fallback demo detector used only when no real detector loads.",
            "detector_mode": "mock",
            "weights": None,
            "detections": detections,
            "has_animal": has_animal,
            "processing_time_ms": 0.0,
        }

    @property
    def is_loaded(self) -> bool:
        return any(detector.is_loaded for detector in self.detectors.values())


detector_instance: Optional[MultiDetectorManager] = None


def get_detector() -> MultiDetectorManager:
    """Get or create the multi-detector manager singleton."""
    global detector_instance
    if detector_instance is None:
        detector_instance = MultiDetectorManager()
    return detector_instance
