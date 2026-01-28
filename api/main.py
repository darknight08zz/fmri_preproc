import sys
import os

# Add parent dir to path so we can import fmri_preproc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import datasets, validation_router, conversion

app = FastAPI(
    title="fMRI Preproc API",
    description="Backend API for fMRI Preprocessing Pipeline",
    version="0.1.0"
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

app.include_router(datasets.router)
app.include_router(validation_router.router)
app.include_router(conversion.router)
from api.routers import pipeline
app.include_router(pipeline.router)

# Mount converted data for static access
# This assumes 'converted_data' is in the project root
static_dir = os.path.join(os.getcwd(), "converted_data")
os.makedirs(static_dir, exist_ok=True)
app.mount("/files", StaticFiles(directory=static_dir), name="files")


@app.get("/")
async def root():
    return {"message": "fMRI Preprocessing API is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

