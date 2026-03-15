from fastapi import APIRouter, Depends  # pyre-ignore
from sqlalchemy.orm import Session  # pyre-ignore
from sqlalchemy import text  # pyre-ignore

from database import get_db
from model_loader import get_model, SortIQModel
from schemas import HealthResponse
from logger import logger  # pyre-ignore

router = APIRouter()

@router.get("", response_model=HealthResponse)
async def check_health(
    db: Session = Depends(get_db),
    model_instance: SortIQModel = Depends(get_model)
):
    """
    Endpoint for cloud health checks (Render, etc.)
    """
    # Check DB Connection
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"Healthcheck DB failure: {str(e)}")
        
    # Check Model loaded
    from model_loader import ensure_models_loaded, get_model
    import os
    ensure_models_loaded()
    model_instance = get_model()
    
    model_loaded = False
    if model_instance and model_instance.model is not None:
         model_loaded = True
    else:
        logger.error("Healthcheck Model failure: Model not loaded in memory")
        
    status = "ok" if (db_connected and model_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        db_connected=db_connected
    )
