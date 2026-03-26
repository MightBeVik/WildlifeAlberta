"""
Configuration settings for the Wildlife Detection & Classification system.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project root
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MEGADETECTOR_MODEL = os.getenv("MEGADETECTOR_MODEL", "yolov5s")  # Default to yolov5s
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))

# Image settings
MAX_IMAGE_SIZE = 1280  # Max dimension for processing
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Alberta Wildlife Species Labels
SPECIES_LABELS = [
    "White-tailed Deer",
    "Mule Deer",
    "Moose",
    "Elk",
    "Black Bear",
    "Grizzly Bear",
    "Gray Wolf",
    "Coyote",
    "Red Fox",
    "Cougar",
    "Lynx",
    "Bobcat",
    "Snowshoe Hare",
    "Red Squirrel",
    "Gray Squirrel",
    "Beaver",
    "Porcupine",
    "Raccoon",
    "Skunk",
    "Bald Eagle",
    "Great Horned Owl",
    "Wild Turkey",
    "Canada Goose",
    "Unknown",
]

# ImageNet class IDs that map to our wildlife species (approximate mapping)
# This maps ImageNet class indices to our SPECIES_LABELS indices
IMAGENET_TO_WILDLIFE = {
    353: 0,   # gazelle -> White-tailed Deer (closest match)
    354: 0,   # impala -> White-tailed Deer
    355: 1,   # ibex -> Mule Deer (closest match)
    346: 2,   # water buffalo -> Moose (closest approximation)
    351: 2,   # ram -> Moose
    295: 4,   # American black bear
    296: 5,   # brown bear -> Grizzly Bear
    269: 7,   # timber wolf -> Gray Wolf  
    270: 7,   # white wolf -> Gray Wolf
    271: 8,   # red fox
    272: 8,   # kit fox -> Red Fox
    286: 9,   # cougar
    287: 10,  # lynx
    335: 16,  # fox squirrel -> Red Squirrel
    336: 15,  # marmot -> Gray Squirrel (closest)
    337: 17,  # beaver
    334: 18,  # porcupine -> Porcupine
    22: 19,   # bald eagle
}

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
