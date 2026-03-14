from fastapi import APIRouter, Depends  # pyre-ignore
from pydantic import BaseModel  # pyre-ignore
from typing import Dict, Any, Optional

from model_loader import get_model, SortIQModel  # pyre-ignore
from logger import logger  # pyre-ignore

router = APIRouter()

class SettingsUpdate(BaseModel):
    threshold: float
    # Future proofing
    binMapping: Optional[Dict[str, Any]] = None 

@router.post("")
async def update_settings(
    settings: SettingsUpdate,
    model: SortIQModel = Depends(get_model)
):
    """
    Update global model settings (e.g. confidence threshold).
    """
    try:
        model.threshold = settings.threshold
        if settings.binMapping:
            # We can optionally overwrite the backend's internal mapping
            model.bin_mapping = settings.binMapping
            
        logger.info(f"Settings updated: threshold={model.threshold}")
        return {"status": "success", "message": "Settings updated", "threshold": model.threshold}
    except Exception as e:
        logger.error(f"Failed to update settings: {str(e)}")
        return {"status": "error", "message": str(e)}
