
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
import logging
from fmri_preproc.utils.dicom import DicomConverter
from fmri_preproc.utils.dicom_pipeline import DicomPipeline

router = APIRouter(prefix="/convert", tags=["conversion"])
logger = logging.getLogger(__name__)

# Global progress store (In-memory for simplicity)
# Key: subject_id (or a unique task ID), Value: Dict with status
pipeline_progress: Dict[str, Dict] = {}

class DicomConversionRequest(BaseModel):
    input_dir: str
    subject: str
    session: Optional[str] = None
    output_dir: Optional[str] = None 
    use_pipeline: bool = False # Toggle for new pipeline

def run_conversion_background(input_dir: str, output_dir: str, subject: str, session: Optional[str]):
    """Legacy dcm2niix wrapper"""
    converter = DicomConverter()
    success, msg = converter.run(input_dir, output_dir, subject, session)
    # No detailed progress for this one, just efficient C++ conversion
    if success:
        logger.info(f"dcm2niix Success: {msg}")
    else:
        logger.error(f"dcm2niix Failed: {msg}")

def run_pipeline_background(input_dir: str, output_dir: str, subject: str, session: Optional[str]):
    """Python-native 11-stage pipeline"""
    pipeline = DicomPipeline(input_dir, subject, session)
    try:
        pipeline_progress[subject] = {"status": "running", "stage": "Initializing", "percent": 0}
        
        for stage_name, percent in pipeline.run(output_dir):
            pipeline_progress[subject] = {
                "status": "running",
                "stage": stage_name,
                "percent": percent
            }
            logger.info(f"Pipeline [{subject}]: {stage_name} ({percent}%)")
            
        pipeline_progress[subject] = {
            "status": "completed",
            "stage": "Complete",
            "percent": 100,
            "output_path": os.path.join(output_dir, f"{subject}_task-rest_bold.nii.gz") # Approximation
        }
    except Exception as e:
        logger.exception(f"Pipeline failed for {subject}")
        pipeline_progress[subject] = {
            "status": "failed",
            "error": str(e),
            "percent": 0
        }

@router.post("/dicom")
def convert_dicom(request: DicomConversionRequest, background_tasks: BackgroundTasks):
    """
    Trigger a DICOM to NIfTI conversion. 
    Use 'use_pipeline=True' for the detailed Python 11-stage pipeline.
    """
    if not os.path.exists(request.input_dir):
        raise HTTPException(status_code=404, detail=f"Input directory not found: {request.input_dir}")

    # Enforce sub- prefix for BIDS compliance
    if not request.subject.startswith("sub-"):
        request.subject = f"sub-{request.subject}"

    # Determine output directory
    output_dir = request.output_dir
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "converted_data", request.subject)
        if request.session:
            output_dir = os.path.join(output_dir, request.session)
    
    if request.use_pipeline:
        # Check if already running?
        if request.subject in pipeline_progress and pipeline_progress[request.subject]['status'] == 'running':
             return {"status": "ignored", "message": f"Pipeline already running for {request.subject}"}

        background_tasks.add_task(
            run_pipeline_background, 
            request.input_dir, 
            output_dir, 
            request.subject, 
            request.session
        )
        return {
            "status": "started", 
            "mode": "python_pipeline",
            "message": f"Detailed pipeline started for {request.subject}",
            "target_output_dir": output_dir
        }
    else:
        # Legacy dcm2niix
        converter = DicomConverter()
        if not converter.check_installed():
             raise HTTPException(status_code=500, detail="dcm2niix is not installed. Use 'use_pipeline=True' to use the Python fallback.")
        
        background_tasks.add_task(
            run_conversion_background, 
            request.input_dir, 
            output_dir, 
            request.subject, 
            request.session
        )
        return {
            "status": "started", 
            "mode": "dcm2niix",
            "message": f"Fast conversion started for {request.subject}",
            "target_output_dir": output_dir
        }

@router.get("/status/{subject}")
def get_conversion_status(subject: str):
    """Get status of the Python pipeline for a subject."""
    status = pipeline_progress.get(subject)
    if not status:
        return {"status": "unknown", "message": "No active pipeline found for this subject."}
    return status
