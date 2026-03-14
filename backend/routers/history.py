from fastapi import APIRouter, Depends, Query, HTTPException  # pyre-ignore
from sqlalchemy.orm import Session  # pyre-ignore
from typing import List, Optional

from database import get_db, WasteScan  # pyre-ignore
from schemas import ScanHistory  # pyre-ignore
from logger import logger  # pyre-ignore

router = APIRouter()

@router.get("", response_model=List[ScanHistory])
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    class_filter: Optional[str] = None,
    interaction_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Endpoint for returning scan history with pagination and filtering.
    """
    try:
        query = db.query(WasteScan).order_by(WasteScan.timestamp.desc())
        
        if class_filter:
            query = query.filter(WasteScan.predicted_class == class_filter)
        if interaction_filter:
            query = query.filter(WasteScan.interaction_type == interaction_filter)
            
        scans = query.offset(offset).limit(limit).all()
        return scans
        
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return []

@router.delete("")
async def clear_all_history(db: Session = Depends(get_db)):
    """
    Clears all scan history.
    """
    try:
        db.query(WasteScan).delete()
        db.commit()
        return {"message": "All history cleared"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing history: {str(e)}")
        from fastapi import HTTPException  # pyre-ignore
        raise HTTPException(status_code=500, detail="Failed to clear history")

@router.delete("/{scan_id}")
async def delete_history_item(scan_id: str, db: Session = Depends(get_db)):
    """
    Deletes a specific scan by ID.
    """
    try:
        scan = db.query(WasteScan).filter(WasteScan.id == scan_id).first()
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        db.delete(scan)
        db.commit()
        return {"message": f"Scan {scan_id} deleted"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting scan {scan_id}: {str(e)}")
        from fastapi import HTTPException  # pyre-ignore
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail="Failed to delete scan")
        raise e


@router.get("/{scan_id}", response_model=ScanHistory)
async def get_history_item(scan_id: str, db: Session = Depends(get_db)):
    """
    Returns details for a single scan.
    """
    try:
        scan = db.query(WasteScan).filter(WasteScan.id == scan_id).first()
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
        return scan
    except Exception as e:
        logger.error(f"Error retrieving scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve scan")
