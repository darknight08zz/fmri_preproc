from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os

# Ensure we can import the core package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fmri_preproc.io.bids import BIDSDataset

router = APIRouter(prefix="/datasets", tags=["datasets"])

class DatasetConfig(BaseModel):
    path: str

@router.get("/available")
async def list_available_datasets():
    """
    List potential BIDS datasets found in common locations.
    """
    datasets = []
    
    # Define search paths relative to project root (../../ from here)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    search_paths = ["example_bids", "converted_data", "derivatives"]
    
    for p in search_paths:
        full_path = os.path.join(root_dir, p)
        if os.path.exists(full_path) and os.path.isdir(full_path):
             # Check if it looks like a dataset (has sub- folders or dataset_description.json)
             # Simple heuristic: treat the folder itself as a dataset
             datasets.append({"name": p, "path": p, "absolute_path": full_path})
             
             # Also check subdirectories of 'converted_data' as they are essentially datasets per subject/session if not merged
             # Actually, converted_data is usually the root for BIDS, containing sub-X folders. 
             # So 'converted_data' IS the dataset path.
    
    return {"datasets": datasets}

@router.post("/subjects")
async def list_subjects(config: DatasetConfig):
    try:
        ds = BIDSDataset(config.path)
        return {"subjects": ds.get_subjects()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scans")
async def list_scans(config: DatasetConfig, subject: str):
    try:
        ds = BIDSDataset(config.path)
        return ds.get_scans(subject)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from datetime import datetime
from fastapi import UploadFile, File

@router.post("/upload")
async def upload_dataset(files: List[UploadFile] = File(...)):
    """
    Handle file uploads.
    Saves files to 'uploads/YYYYMMDD_HHMMSS' and returns the path.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../uploads", timestamp))
        os.makedirs(upload_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            # Ensure subdirs if filename has paths (unlikely for simple upload but good practice)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(file.filename)
            
        return {"path": upload_dir, "message": f"Uploaded {len(saved_files)} files"}
        
    except Exception as e:
        print(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import httpx
class UrlConfig(BaseModel):
    url: str
    filename: str = "downloaded_file.nii.gz"

@router.post("/upload-url")
async def download_from_url(config: UrlConfig):
    """
    Download file from a URL.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../uploads", timestamp))
        os.makedirs(upload_dir, exist_ok=True)
        
        # Determine filename if not provided or generic
        fname = config.filename
        if not fname or fname == "string":
             fname = config.url.split("/")[-1]
             if not fname: fname = "downloaded_file"
        
        file_path = os.path.join(upload_dir, fname)
        
        print(f"Downloading {config.url} to {file_path}")
        
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', config.url, follow_redirects=True) as response:
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                        
        return {"path": upload_dir, "message": f"Downloaded {fname}"}

    except Exception as e:
        print(f"URL Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


