
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
import logging
import traceback

# Import the new Controller
from fmri_preproc.core.controller import PipelineController

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)

# Global progress store
# Structure: { subject: { status: str, message: str, stage: str, output_path: str, logs: list } }
pipeline_state: Dict[str, Dict] = {}

class PipelineRunRequest(BaseModel):
    subject: str
    bids_dir: str # Path to the BIDS dataset root
    output_dir: Optional[str] = None # Defaults to project_root/derivatives

def execute_pipeline(subject: str, bids_dir: str, output_dir: str):
    try:
        pipeline_state[subject] = {
            "status": "running", 
            "stage": "Initializing", 
            "message": "Starting pipeline...",
            "logs": []
        }
        
        # Determine output root if not provided
        if not output_dir:
            # Assuming project root is 2 levels up from api/routers/pipeline.py -> api/routers -> api -> root
            # Actually, standard is usually explicitly passed or config.
            # Let's derive from current CWD or bids_dir sibling.
            # Default: <bids_dir>/derivatives
            # OR local 'derivatives'
            output_dir = os.path.join(os.getcwd(), "derivatives")
            
        pipeline_state[subject]["message"] = f"Output root: {output_dir}"
        
        # Initialize Controller
        controller = PipelineController(output_root=output_dir)
        
        # We want to capture logs/print statments from the controller?
        # For now, we rely on the Controller running synchronously in this thread.
        # Ideally, we'd hook into the Node execution to update status.
        # But `wf.run()` is blocking.
        
        # Run
        pipeline_state[subject]["stage"] = "Execution"
        
        def update_progress(node_name, status):
            if status == "running":
                msg = f"Running {node_name}..."
                pipeline_state[subject]["stage"] = f"Running {node_name}"
                pipeline_state[subject]["message"] = msg
                pipeline_state[subject]["logs"].append(msg) # Now enabled for immediate feedback
            elif status == "completed":
                msg = f"Completed {node_name}."
                pipeline_state[subject]["message"] = msg
                pipeline_state[subject]["logs"].append(msg)
        
        controller.run(subject, bids_dir, status_callback=update_progress)
        
        pipeline_state[subject] = {
            "status": "completed",
            "stage": "Finished",
            "message": "Pipeline execution completed successfully.",
            "output_path": os.path.join(output_dir, subject),
            "logs": ["Refer to QC report for details."]
        }
        
    except Exception as e:
        logger.exception(f"Preprocessing failed for {subject}")
        pipeline_state[subject] = {
            "status": "failed",
            "stage": "Error",
            "message": str(e),
            "logs": [traceback.format_exc()]
        }

@router.post("/run")
async def run_preprocessing(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    Start the fMRI Preprocessing Pipeline for a given subject.
    """
    subject = request.subject
    # Normalize subject ID
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"
        
    if not os.path.exists(request.bids_dir):
         raise HTTPException(status_code=404, detail=f"BIDS directory not found: {request.bids_dir}")
         
    if subject in pipeline_state and pipeline_state[subject]['status'] == 'running':
        return {"status": "ignored", "message": f"Pipeline already running for {subject}"}
        
    background_tasks.add_task(
        execute_pipeline,
        subject,
        request.bids_dir,
        request.output_dir
    )
    
    return {"status": "started", "message": f"Preprocessing started for {subject}"}

@router.get("/status/{subject}")
async def get_status(subject: str):
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}" # Try both
        
    state = pipeline_state.get(subject)
    if not state:
        # Check raw input if normalization failed match
        state = pipeline_state.get(request.subject) if 'request' in locals() else None
        
    if not state:
        return {"status": "unknown", "message": "No pipeline history for this subject."}
        
    return state
