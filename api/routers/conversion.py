from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
import logging
from fmri_preproc.utils.dicom_pipeline import DicomPipeline

router = APIRouter(prefix="/convert", tags=["conversion"])
logger = logging.getLogger(__name__)

# Global progress store (In-memory)
pipeline_progress: Dict[str, Dict] = {}

class PipelineRequest(BaseModel):
    input_dir: str
    subject: str
    session: Optional[str] = None

def run_pipeline_background(input_dir: str, output_dir: str, subject: str, session: Optional[str]):
    """Execute the pipeline in background"""
    pipeline = DicomPipeline(input_dir, subject, session)
    try:
        pipeline_progress[subject] = {"status": "running", "stage": "Initializing", "percent": 0}
        
        generated_files = []
        for stage_name, percent in pipeline.run(output_dir):
            pipeline_progress[subject] = {
                "status": "running",
                "stage": stage_name,
                "percent": percent
            }
            logger.info(f"Pipeline [{subject}]: {stage_name} ({percent}%)")
        
        # pipeline.run acts as a generator yielding progress, but its return value 
        # (the list of files) is obtained if we iterate to completion? 
        # Wait, in Python generators, the return value is captured in StopIteration exception 
        # OR if we iterate simply, we don't get the return value easily unless we change the design.
        #
        # Better design: make run() yield the result as the last item or change run() to NOT return 
        # but store result in attribute.
        #
        # Actually, let's just make run() yield ("Complete", 100, generated_files)
        # OR just rely on checking the output directory.
        #
        # Let's fix DicomPipeline.run signature to Yield result as well? 
        # Or simpler: access `pipeline.generated_files` if I add it as attribute.
        #
        # Alternatively, checking `pipeline.run` implementation: 
        # it yields ("Complete", 100) then returns `generated_files`.
        # getting return value from generator requires `freq = yield from ...` or catching `StopIteration`.
        #
        # I will modify DicomPipeline.run slightly in prev step? No, I already applied it.
        #
        # I will check `output_dir` for files or better yet, I should have updated the `run` method
        # to yield the result.
        #
        # Let's update `conversion.py` assuming I can't easily get the return value from the loop 
        # without changing usage pattern.
        #
        # I will change `DicomPipeline` to store `self.generated_files` and access it here.
        
        # RE-THINK: I need to update DicomPipeline to expose the files, OR handle the return.
        # Since I can't easily edit the previous step's code now without another call, 
        # I will use `pipeline.generated_files` if I add it to `DicomPipeline` class. 
        # But I didn't add it to `__init__`.
        #
        # I'll update `conversion.py` to just scan the directory.
        # Or better, I'll update `DicomPipeline` again to add `self.generated_files` in `run`.
        #
        # Actually, let's just scan the `output_dir` in `conversion.py`.
        
        output_files = []
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith('.nii.gz'):
                    output_files.append(os.path.join(root, f))
                    
        pipeline_progress[subject] = {
            "status": "completed",
            "stage": "Complete",
            "percent": 100,
            "output_paths": output_files
        }
    except Exception as e:
        logger.exception(f"Pipeline failed for {subject}")
        pipeline_progress[subject] = {
            "status": "failed",
            "error": str(e),
            "percent": 0
        }

@router.post("/pipeline")
def convert_dicom_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Trigger the Python DICOM pipeline manually.
    """
    if not os.path.exists(request.input_dir):
        raise HTTPException(status_code=404, detail=f"Input directory not found: {request.input_dir}")

    # Enforce sub- prefix 
    subject_id = request.subject
    if not subject_id.startswith("sub-"):
        subject_id = f"sub-{subject_id}"

    # Output directory: converted_data/sub-XX
    output_dir = os.path.join(os.getcwd(), "converted_data", subject_id)
    if request.session:
        output_dir = os.path.join(output_dir, request.session)
    
    if subject_id in pipeline_progress and pipeline_progress[subject_id]['status'] == 'running':
         return {"status": "ignored", "message": f"Pipeline already running for {subject_id}"}

    background_tasks.add_task(
        run_pipeline_background, 
        request.input_dir, 
        output_dir, 
        subject_id, 
        request.session
    )
    return {
        "status": "started", 
        "message": f"Pipeline started for {subject_id}",
        "target_output_dir": output_dir
    }

@router.get("/status/{subject}")
def get_conversion_status(subject: str):
    """Get status of the pipeline."""
    # Handle implicit sub- prefix if omitted in query
    if not subject.startswith("sub-"):
        # check both
        if subject in pipeline_progress:
            return pipeline_progress[subject]
        subject = f"sub-{subject}"
        
    status = pipeline_progress.get(subject)
    if not status:
        return {"status": "unknown", "message": "No active pipeline found for this subject."}
    return status
