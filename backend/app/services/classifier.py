"""
Species classifier service using EfficientNet-B3 or mock fallback.
Stage 2 of the two-stage pipeline: Classifies detected animal crops into species.
"""
import time
import random
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from app.config import SPECIES_LABELS, IMAGENET_TO_WILDLIFE

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
    logger.info("PyTorch loaded successfully")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — using mock classifier")


class WildlifeClassifier:
    """
    Species classification using EfficientNet-B3 pretrained on ImageNet.
    Maps ImageNet predictions to Alberta wildlife species.
    Falls back to mock classifications if PyTorch is not available.
    
    STOPGAP MODEL — Replace with custom-trained model later:
        1. Train your own model on the Alberta camera trap dataset
        2. Save weights to data/models/custom_classifier.pt
        3. Update _load_model() to load your custom architecture
        4. Update SPECIES_LABELS in config.py with your training classes
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load the EfficientNet-B3 classification model."""
        if not TORCH_AVAILABLE:
            logger.info("Running in MOCK mode — no classification model loaded")
            return

        try:
            # Load EfficientNet-B3 pretrained on ImageNet
            self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.model.eval()

            # Standard ImageNet preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self.model_loaded = True
            logger.info("EfficientNet-B3 classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            self.model_loaded = False

    def classify(self, image: Image.Image) -> dict:
        """
        Classify a cropped animal image into a species.
        
        Args:
            image: PIL Image of the cropped animal detection
            
        Returns:
            dict with keys: species, confidence, top_5
        """
        start = time.time()

        if self.model_loaded and self.model is not None:
            result = self._real_classify(image)
        else:
            result = self._mock_classify()

        elapsed = (time.time() - start) * 1000
        result["processing_time_ms"] = round(elapsed, 2)
        return result

    def _real_classify(self, image: Image.Image) -> dict:
        """Run real EfficientNet classification."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 20)

        # Map ImageNet classes to wildlife species
        wildlife_scores = {}
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            wildlife_idx = IMAGENET_TO_WILDLIFE.get(idx)
            if wildlife_idx is not None:
                species = SPECIES_LABELS[wildlife_idx]
                if species not in wildlife_scores:
                    wildlife_scores[species] = 0
                wildlife_scores[species] += prob

        # If no mapped species found, use "Unknown"
        if not wildlife_scores:
            wildlife_scores["Unknown"] = 1.0

        # Sort by confidence
        sorted_species = sorted(wildlife_scores.items(), key=lambda x: x[1], reverse=True)

        # Build top 5
        top_5 = [
            {"species": species, "confidence": round(conf, 4)}
            for species, conf in sorted_species[:5]
        ]

        # Normalize confidences to sum to 1
        total_conf = sum(item["confidence"] for item in top_5)
        if total_conf > 0:
            for item in top_5:
                item["confidence"] = round(item["confidence"] / total_conf, 4)

        return {
            "species": top_5[0]["species"] if top_5 else "Unknown",
            "confidence": top_5[0]["confidence"] if top_5 else 0.0,
            "top_5": top_5,
        }

    def _mock_classify(self) -> dict:
        """
        Generate mock classification for demo/testing.
        Simulates realistic Alberta wildlife species distribution.
        """
        # Weighted species distribution typical for Alberta camera traps
        species_weights = {
            "White-tailed Deer": 0.25,
            "Mule Deer": 0.10,
            "Moose": 0.08,
            "Elk": 0.07,
            "Black Bear": 0.10,
            "Grizzly Bear": 0.03,
            "Gray Wolf": 0.04,
            "Coyote": 0.12,
            "Red Fox": 0.05,
            "Cougar": 0.02,
            "Lynx": 0.02,
            "Snowshoe Hare": 0.04,
            "Red Squirrel": 0.05,
            "Porcupine": 0.02,
            "Raccoon": 0.01,
        }

        species_list = list(species_weights.keys())
        weights = list(species_weights.values())

        # Pick primary species
        primary = random.choices(species_list, weights=weights, k=1)[0]
        primary_conf = round(random.uniform(0.55, 0.95), 4)

        # Generate top 5 with decreasing confidence
        remaining = [s for s in species_list if s != primary]
        random.shuffle(remaining)
        remaining_conf = 1.0 - primary_conf

        top_5 = [{"species": primary, "confidence": primary_conf}]
        for i, species in enumerate(remaining[:4]):
            conf = round(remaining_conf * random.uniform(0.1, 0.5), 4)
            remaining_conf -= conf
            top_5.append({"species": species, "confidence": max(0.01, conf)})

        return {
            "species": primary,
            "confidence": primary_conf,
            "top_5": top_5,
        }

    @property
    def is_loaded(self) -> bool:
        return self.model_loaded


# Singleton instance
classifier_instance: Optional[WildlifeClassifier] = None


def get_classifier() -> WildlifeClassifier:
    """Get or create the singleton classifier instance."""
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = WildlifeClassifier()
    return classifier_instance
