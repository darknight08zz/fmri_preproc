# Codebase Summary & Project Overview

## 1. Project Goal: What are we doing?
We are building a **Full-Stack fMRI Preprocessing Application**. 
The goal is to provide a user-friendly interface for researchers to preprocess fMRI data using a standardized, scientific pipeline (BIDS-compatible). The system automates complex neuroimaging tasks like motion correction, distortion correction, and normalization, delivering "analysis-ready" data.

## 2. Architecture: How are we doing it?
The project follows a modern 3-tier architecture:

### A. Frontend (UI)
- **Tech Stack**: Next.js 16 (React 19), TailwindCSS 4, Lucide React.
- **Location**: `web/`
- **Role**: 
  - Provides a dashboard for users.
  - Allows selecting BIDS datasets.
  - Triggers preprocessing runs for specific subjects.
  - (Planned) Visualizing QC reports and logs.

### B. Backend (API)
- **Tech Stack**: FastAPI (Python).
- **Location**: `api/`
- **Role**:
  - Serves as the bridge between UI and the processing core.
  - **Endpoints**:
    - `POST /pipeline/run`: Accepts `{bids_path, subject}` and starts processing.
    - `POST /convert/dicom`: Accepts `{input_dir, subject}` and starts DICOM->NIfTI conversion.
    - `GET /datasets`: Lists available datasets (implied).
  - **Execution**: Uses `BackgroundTasks` to run the heavy preprocessing pipeline asynchronously so the UI doesn't freeze.

### C. Core Logic (Processing Engine)
- **Tech Stack**: Python, Nibabel, NumPy.
- **Location**: `fmri_preproc/`
- **Key Components**:
  - **`PipelineManager`** (`fmri_preproc/core/manager.py`): The conductor. It defines the sequence of steps and manages data flow between them.
  - **Modules**:
    - `utils/`: Utilities (DICOM conversion).
    - `anat/`: Anatomical processing (Bias correction, Skull stripping, Segmentation, Normalization).
    - `func/`: Functional processing (Slice timing, Motion correction, SDC, Coregistration).
    - `confounds/`: Noise estimation (Motion metrics, ACompCor).
    - `qc/`: Quality Control report generation.

## 3. The Preprocessing Pipeline
The pipeline is implemented in `PipelineManager.run_subject`. It processes one subject at a time.

**Key Features:**
- **Single-Shot Resampling**: To minimize interpolation artifacts, it calculates multiple transforms (Motion, SDC, Coregistration, Normalization) separately but applies them all in one single resampling step (`SpatialTransforms`).
- **BIDS Compatible**: It expects data in the Brain Imaging Data Structure (BIDS) format.

**Detailed Stages:**
1.  **Anatomical (T1w)**:
    - Bias Field Correction.
    - Skull Stripping (Brain extraction).
    - Segmentation (WM/GM/CSF).
    - Normalization: Computes transform from T1 space to MNI standard space.

2.  **Functional (BOLD)**:
    - **Dummy Scans**: Removes initial unsteady volumes.
    - **Slice Timing Correction**: Corrects for acquisition time differences between slices.
    - **Motion Correction (Estimate)**: Estimates head motion parameters (`.par`, `.mat`).
    - **Susceptibility Distortion Correction (SDC)**: Estimates warp field to fix geometric distortions.
    - **Coregistration**: Computes transform from Functional (Mean EPI) to Anatomical (T1).
    - **Resampling**: Applies `[T1->MNI] + [EPI->T1] + [SDC] + [Motion]` transforms to the original BOLD data to move it to MNI space in one go.

3.  **Confounds & Post-processing**:
    - **Spatial Smoothing**: Applies Gaussian smoothing (6mm FWHM) on the MNI-space data.
    - **Temporal Filtering**: High-pass filtering (0.01 Hz).
    - **Motion Metrics**: Calculates Framewise Displacement (FD).
    - **Scaling**: Intensity scaling on the output.

4.  **Quality Control (QC)**:
    - Generates an HTML report summarizing the run.

## 4. Directory Structure
```
.
├── api/                  # FastAPI Backend
│   ├── main.py           # App entry point
│   └── routers/          # API endpoints
├── web/                  # Next.js Frontend
│   ├── package.json
│   └── src/              # React components
├── fmri_preproc/         # Core Python Package
│   ├── core/             # Pipeline orchestration (manager.py)
│   ├── anat/             # Anatomical modules
│   ├── func/             # Functional modules
│   └── ...
└── derivatives/          # (Default) Output folder for processed data
```
