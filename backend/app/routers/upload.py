"""
Upload router — handles image uploads and image listing.
"""
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS
from app.services.preprocessor import generate_image_id, get_image_info
from app.models.schemas import UploadResponse, BatchUploadResponse, ImageInfo

router = APIRouter(prefix="/api", tags=["upload"])

# In-memory store (replace with database in production)
uploaded_images: dict = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload a single image for processing."""
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}",
        )

    # Generate unique ID and save
    image_id = generate_image_id()
    filename = f"{image_id}{ext}"
    filepath = UPLOAD_DIR / filename

    try:
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Get image info
    info = get_image_info(filepath)
    if "error" in info:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid image: {info['error']}")

    # Store metadata
    uploaded_images[image_id] = {
        "image_id": image_id,
        "filename": file.filename,
        "saved_as": filename,
        "filepath": str(filepath),
        "size_bytes": info["size_bytes"],
        "width": info["width"],
        "height": info["height"],
        "upload_time": datetime.now().isoformat(),
    }

    return UploadResponse(
        image_id=image_id,
        filename=file.filename,
        filepath=f"/uploads/{filename}",
        size_bytes=info["size_bytes"],
        width=info["width"],
        height=info["height"],
        message="Image uploaded successfully",
    )


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple images at once."""
    uploaded = []
    failed = []

    for file in files:
        try:
            result = await upload_image(file)
            uploaded.append(result)
        except HTTPException as e:
            failed.append({"filename": file.filename, "error": e.detail})
        except Exception as e:
            failed.append({"filename": file.filename, "error": str(e)})

    return BatchUploadResponse(
        uploaded=uploaded,
        failed=failed,
        total=len(files),
    )


@router.get("/images")
async def list_images():
    """List all uploaded images."""
    return {
        "images": list(uploaded_images.values()),
        "total": len(uploaded_images),
    }


@router.get("/images/{image_id}")
async def get_image(image_id: str):
    """Get details of a specific uploaded image."""
    if image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")
    return uploaded_images[image_id]


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """Delete an uploaded image."""
    if image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")

    info = uploaded_images[image_id]
    filepath = Path(info["filepath"])
    filepath.unlink(missing_ok=True)
    del uploaded_images[image_id]

    return {"message": f"Image {image_id} deleted", "image_id": image_id}


def get_uploaded_images():
    """Access the uploaded images store from other modules."""
    return uploaded_images
