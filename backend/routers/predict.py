import os
import time
from fastapi import APIRouter, File, UploadFile, Depends, Request, HTTPException, Security  # pyre-ignore
from fastapi.security.api_key import APIKeyHeader  # pyre-ignore
from sqlalchemy.orm import Session  # pyre-ignore
from typing import List, Dict, Optional
from slowapi import Limiter  # pyre-ignore
from slowapi.util import get_remote_address  # pyre-ignore
import io
import cv2  # pyre-ignore
import base64
import numpy as np  # pyre-ignore
from PIL import Image  # pyre-ignore

from database import get_db, WasteScan  # pyre-ignore
from schemas import RealtimePredictRequest, RealtimePredictResponse  # pyre-ignore
from model_loader import get_model, SortIQModel, THRESHOLDS  # pyre-ignore
from logger import logger  # pyre-ignore
from preprocessing import generate_thumbnail  # pyre-ignore

# Only these classes should be stored in history
VALID_CLASSES = {"plastic", "paper", "metal", "glass"}

# Global state for debouncing
_last_save_time: Dict[str, float] = {}
_last_frame_mean = None
_processing = False  # Busy-lock to prevent queue buildup

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# API Key Security
API_KEY = os.getenv("API_KEY", "test_key")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if not API_KEY or api_key_header == API_KEY:
        return api_key_header
    if api_key_header is not None:
        raise HTTPException(status_code=403, detail="Invalid API KEY")
    return None

def robot_painter(img_pil: Image.Image, detections: List[dict]) -> str:
    """Robot Painter: Draws tech-circles and labels on thumbnails."""
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx = int(x1 + (x2 - x1) / 2)
        cy = int(y1 + (y2 - y1) / 2)
        radius = int(max(x2 - x1, y2 - y1) / 2)
        
        hex_color = det["box_color_hex"].lstrip('#')
        bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
        
        # Draw tech-circle
        cv2.circle(img_cv, (cx, cy), radius, bgr_color, 2)
        if det.get("is_waste"):
            cv2.circle(img_cv, (cx, cy), radius + 4, bgr_color, 1) # Double ring for waste
        
        # Label
        label_text = f"{det['label']} {int(det['confidence']*100)}%"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_cv, (cx - w//2, cy - radius - 25), (cx + w//2, cy - radius - 5), bgr_color, -1)
        cv2.putText(img_cv, label_text, (cx - w//2, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    _, buffer = cv2.imencode('.jpg', img_cv)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def robot_recorder(db: Session, best_det: dict, image_bytes: bytes) -> Optional[str]:
    """Robot Recorder: Saves to the database."""
    raw_label = (best_det.get("raw_label") or best_det.get("label", "Unknown")).lower().strip()
    conf = best_det.get("confidence", 0)
    itype = best_det.get("interaction_type", "waste")
    if itype == "waste" and raw_label not in VALID_CLASSES:
        itype = "interaction"
        
    try:
        thumbnail_url = generate_thumbnail(image_bytes, box=best_det.get("box"))
        db_scan = WasteScan(
            predicted_class=raw_label,
            confidence=conf,
            image_thumbnail_url=thumbnail_url,
            interaction_type=itype,
            robot_message=best_det.get("message", None)
        )
        db.add(db_scan)
        db.commit()
        db.refresh(db_scan)
        return db_scan.id
    except Exception as e:
        logger.error(f"Save failed: {e}")
        db.rollback()
        return None

def determine_scene_state(detections: List[dict]) -> str:
    if not detections:
        return "empty"
    if any(d["is_waste"] for d in detections):
        return "waste_found"
    return "no_waste"

def generate_summary(detections: List[dict]) -> str:
    if not detections:
        return "Scout: Scene clear."
    
    waste_items = [d['label'] for d in detections if d['is_waste']]
    if waste_items:
        return f"Analyst: {len(waste_items)} waste source(s) identified."
    return "Scout: Non-recyclables detected."

@router.post("-realtime", response_model=RealtimePredictResponse)
async def predict_realtime(
    req: RealtimePredictRequest,
    request: Request,
    model: SortIQModel = Depends(get_model)
):
    """
    Robot Team Realtime Pipeline (Preview Only).
    No database writes allowed here.
    """
    global _last_frame_mean, _processing
    
    from model_loader import ensure_models_loaded
    ensure_models_loaded()
    
    if _processing:
        return RealtimePredictResponse(detections=[], summary="Busy", scene_state="skipped")
    _processing = True
    
    try:
        b64_data = req.frame_base64
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]
            
        image_bytes = base64.b64decode(b64_data)
        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
            
        # Frame Deduplication
        img_arr = np.array(img_pil)
        current_mean = np.mean(img_arr)
        if _last_frame_mean is not None:
            if abs(current_mean - _last_frame_mean) / (_last_frame_mean + 1e-5) < 0.015:
                return RealtimePredictResponse(detections=[], summary="Skipped", scene_state="skipped")
        _last_frame_mean = current_mean
        
        # Custom colors from headers
        color_overrides = {}
        for mat in ["Glass", "Plastic", "Metal", "Paper"]:
            val = request.headers.get(f"X-Color-{mat}")
            if val:
                color_overrides[mat] = val

        # Prediction
        logger.info(f"[API] Calling model.predict_scene for REALTIME preview")
        detections = model.predict_scene(img_pil, color_overrides=color_overrides)
        scene_state = determine_scene_state(detections)
        summary = generate_summary(detections)
        
        logger.info(f"[API] Realtime result: {len(detections)} dets, saved=False")
        return RealtimePredictResponse(
            detections=detections,
            summary=summary,
            scene_state=scene_state,
            saved=False,
            saved_id=None,
            auto_saved=False
        )
        
    except Exception as e:
        logger.error(f"Error in predict_realtime: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _processing = False

@router.post("-upload", response_model=RealtimePredictResponse)
async def predict_upload(
    request: Request,
    file: UploadFile = File(...),
    model: SortIQModel = Depends(get_model),
    db: Session = Depends(get_db)
):
    """
    Manual Capture Endpoint.
    This is the ONLY endpoint that saves to the database.
    """
    try:
        from model_loader import ensure_models_loaded
        ensure_models_loaded()
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Custom colors from headers
        color_overrides = {}
        for mat in ["Glass", "Plastic", "Metal", "Paper"]:
            val = request.headers.get(f"X-Color-{mat}")
            if val:
                color_overrides[mat] = val

        detections = model.predict_scene(image, color_overrides=color_overrides)
        best_det = None
        if detections:
            wastes = [d for d in detections if d.get("is_waste")]
            if wastes:
                best_det = max(wastes, key=lambda d: d.get("confidence", 0))
            else:
                best_det = detections[0]

        saved_id = None
        if best_det:
            logger.info(f"[UPLOAD] Saving scan: {best_det.get('label')} {best_det.get('confidence')}")
            saved_id = robot_recorder(db, best_det, bytes(contents))
            logger.info(f"[UPLOAD] Saved with id: {saved_id}")
            
        return RealtimePredictResponse(
            detections=detections,
            summary=generate_summary(detections),
            scene_state=determine_scene_state(detections),
            saved=True if saved_id else False,
            saved_id=saved_id,
            auto_saved=False
        )
    except Exception as e:
        logger.error(f"Error in predict_upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
