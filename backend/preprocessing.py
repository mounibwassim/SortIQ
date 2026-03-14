import io
import os
import uuid
from typing import Optional, List
import numpy as np  # pyre-ignore
from PIL import Image  # pyre-ignore
from fastapi import HTTPException  # pyre-ignore
from logger import logger  # pyre-ignore

# Constants
TARGET_SIZE = (224, 224)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 5 * 1024 * 1024)) # 5MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/jpg"]

THUMBNAIL_DIR = os.path.join("uploads", "thumbnails")
if not os.path.exists(THUMBNAIL_DIR):
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def validate_image_file(content_type: Optional[str], file_size: Optional[int]):
    """
    Validates uploaded file type and size.
    file_size may be None for browser uploads — size is validated separately from actual bytes.
    """
    # Normalize content_type: browsers may send 'image/jpeg; something'
    actual_type = (content_type or "").split(";")[0].strip().lower()
    if actual_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Invalid content type: {content_type}")
        raise HTTPException(status_code=400, detail=f"Invalid image format '{actual_type}'. Supported: JPEG, PNG.")
    
    if file_size is not None and isinstance(file_size, int) and file_size > MAX_IMAGE_SIZE:
        logger.warning(f"File too large: {file_size} bytes")
        raise HTTPException(status_code=400, detail=f"Image too large. Max size is {MAX_IMAGE_SIZE/(1024*1024):.0f}MB.")
    
import cv2  # pyre-ignore

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Prepares image for MobileNetV2 inference matching the exact training pipeline:
    RGB -> resize -> cast to float32 -> normalize by 255.0 -> expand dims.
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (in case of RGBA/Grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Resize to 224x224
        img = img.resize(TARGET_SIZE)
        
        # Convert to numpy array and cast to float32
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] exactly like training
        img_array = img_array / 255.0
        
        # Add batch dimension: (1, 224, 224, 3)
        final_array = np.expand_dims(img_array, axis=0)
        
        return final_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image file. May be corrupted.")


def generate_thumbnail(image_bytes: bytes, box: Optional[List[int]] = None) -> Optional[str]:
    """
    Saves the exact original image at good quality.
    No zoom. No crop. No blur. Full picture.
    Max size 800x800 to keep file size reasonable.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to max 800x800 keeping aspect ratio
        # This keeps the FULL image — no cropping, no zoom
        img.thumbnail((800, 800), Image.LANCZOS)
        
        # Save at high quality
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=92, optimize=True)
        buffer.seek(0)
        
        # Save to static folder
        filename  = f"thumb_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(THUMBNAIL_DIR, filename)
        
        with open(save_path, "wb") as f:
            f.write(buffer.getvalue())
        
        return f"/static/thumbnails/{filename}"
        
    except Exception as e:
        logger.error(f"generate_thumbnail failed: {e}")
        return None
