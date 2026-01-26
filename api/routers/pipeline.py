from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fmri_preproc.core.manager import PipelineManager
import os

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

class RunConfig(BaseModel):
    bids_path: str
    subject: str
    output_path: str = "derivatives"

def _run_pipeline_task(config: RunConfig):
    # Ensure abs paths
    out_dir = os.path.abspath(config.output_path)
    mgr = PipelineManager(output_root=out_dir)
    mgr.run_subject(config.bids_path, config.subject)

@router.post("/run")
async def run_pipeline(config: RunConfig, background_tasks: BackgroundTasks):
    # Validate basics
    if not os.path.exists(config.bids_path):
        raise HTTPException(status_code=404, detail="BIDS path not found")
        
    # Run in background
    background_tasks.add_task(_run_pipeline_task, config)
    
    return {"status": "started", "subject": config.subject, "message": "Pipeline execution started in background"}
