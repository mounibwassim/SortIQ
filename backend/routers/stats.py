from fastapi import APIRouter, Depends  # pyre-ignore
from sqlalchemy.orm import Session  # pyre-ignore
from sqlalchemy import func  # pyre-ignore
from pydantic import BaseModel  # pyre-ignore

from database import get_db, WasteScan  # pyre-ignore
from schemas import StatsResponse  # pyre-ignore
from logger import logger  # pyre-ignore

router = APIRouter()

@router.get("", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """
    Endpoint for system analytics and scan statistics.
    """
    try:
        # Total scans (include EVERYTHING for history/analytics progress)
        total_scans = db.query(WasteScan).count()
        
        # Class distribution
        distribution_query = db.query(
            WasteScan.predicted_class, 
            func.count(WasteScan.id)
        ).filter(WasteScan.interaction_type == "waste").group_by(WasteScan.predicted_class).all()
        
        class_distribution = {row[0]: row[1] for row in distribution_query}
        
        # Average confidence
        avg_conf_query = db.query(func.avg(WasteScan.confidence)).filter(WasteScan.interaction_type == "waste").scalar()
        average_confidence = float(avg_conf_query) if avg_conf_query is not None else 0.0
        
        # Interactions stats
        interaction_query = db.query(
            WasteScan.interaction_type,
            func.count(WasteScan.id)
        ).group_by(WasteScan.interaction_type).all()
        interaction_stats = {row[0]: row[1] for row in interaction_query}
        
        # We hardcode the model accuracy based on Kaggle/TCV notebook results for the demo
        model_accuracy = 0.92
        
        waste_by_class = {
            "Glass":   db.query(WasteScan).filter_by(predicted_class="glass", interaction_type="waste").count(),
            "Metal":   db.query(WasteScan).filter_by(predicted_class="metal", interaction_type="waste").count(),
            "Paper":   db.query(WasteScan).filter_by(predicted_class="paper", interaction_type="waste").count(),
            "Plastic": db.query(WasteScan).filter_by(predicted_class="plastic", interaction_type="waste").count(),
        }
        total_waste_scans = sum(waste_by_class.values())
        
        return StatsResponse(
            total_scans=total_scans,
            model_accuracy=model_accuracy,
            average_confidence=average_confidence,
            class_distribution=class_distribution,
            interaction_stats=interaction_stats,
            waste_by_class=waste_by_class,
            total_waste_scans=total_waste_scans
        )
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        # Return empty safe default
        return StatsResponse(
            total_scans=0,
            model_accuracy=0.92,
            average_confidence=0.0,
            class_distribution={},
            interaction_stats={},
            waste_by_class={
                "Glass": 0,
                "Metal": 0,
                "Paper": 0,
                "Plastic": 0
            },
            total_waste_scans=0
        )
