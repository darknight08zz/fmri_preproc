# fMRI Preprocessing System Overview

This document provides a comprehensive technical summary of the fMRI Preprocessing Application, including its architecture, technology stack, and a detailed breakdown of the preprocessing pipeline.

## 1. System Architecture

The application follows a modern full-stack architecture, designed for local deployment with a focus on privacy and performance.

### **Frontend (Web)**
*   **Framework**: Next.js 14+ (React)
*   **Styling**: TailwindCSS
*   **Visualization**: `niivue` (WebG-based NIfTI rendering)
*   **Communication**: REST API (Axios)

### **Backend (API)**
*   **Framework**: FastAPI (Python)
*   **Asynchronous Model**: Handles long-running pipeline tasks via background threads.
*   **Data Standard**: Fully compliant with **BIDS (Brain Imaging Data Structure)**.
*   **Storage**: Local filesystem storage (organized in `uploads/` and `converted_data/`).

---

## 2. Technology Stack & Tools

The core processing pipeline has been modernized to remove dependencies on legacy C++ tools (like FSL, ANTs, AFNI) where possible, replacing them with **Native Python Implementations**.

| Component | Library / Tool | Description |
| :--- | :--- | :--- |
| **Neuroimaging I/O** | `nibabel` | Reading/Writing NIfTI files and header manipulation. |
| **Numerical Computing** | `numpy`, `scipy` | Matrix operations, interpolation, and optimization algorithms. |
| **Image Processing** | `scipy.ndimage` | 3D convolutions, affine transforms, and resampling. |
| **Machine Learning** | `scikit-learn` | Used for K-Means clustering (Segmentation). |
| **Pipeline Management** | `fmri_preproc.core` | Custom linear workflow manager. |
| **BIDS Handling** | `pybids` | Indexing and querying BIDS datasets. |

### **Native Python vs. Legacy Wrappers**
The system prioritizes native implementations for easier deployment (no need to install FSL/ANTs separately on Windows).

*   **Motion Correction**: Native `scipy.optimize` (Rigid Body 6DOF) replaces FSL MCFLIRT.
*   **Segmentation**: Native `sklearn.KMeans` + MRF Regularization replaces FSL FAST.
*   **Coregistration**: Native Mutual Information optimization replaces FSL FLIRT.
*   **Normalization**: Native Affine + Demons Registration replaces ANTs SyN.
*   **Slice Timing**: Native Cubic Spline Interpolation replaces FSL SliceTimer.

---

## 3. Preprocessing Pipeline (The "11 Stages")

The pipeline uses a **Single-Shot Resampling** approach to minimize interpolation artifacts. Instead of resampling the image at every step (which blurs data), transforms are calculated for each step and combined into a single operation at the end.

### **Phase 1: Anatomical (Anat)**

1.  **Bias Field Correction**
    *   **Goal**: Fix intensity inhomogeneities (shading artifacts) caused by MRI coil sensitivity.
    *   **Method**: Native N4-like algorithm.
    *   **Input**: Raw T1w $\to$ **Output**: Bias-corrected T1w.

2.  **Skull Stripping**
    *   **Goal**: Remove non-brain tissue (skull, neck, eyes).
    *   **Method**: Intensity thresholding + morphological operations.
    *   **Input**: Bias-corrected T1w $\to$ **Output**: Brain-extracted T1w.

3.  **Segmentation**
    *   **Goal**: Classify tissue types (Gray Matter, White Matter, CSF).
    *   **Method**: `sklearn` K-Means Clustering (k=3) with MRF-like spatial regularization.
    *   **Output**: Probability maps for GM, WM, CSF.

4.  **Normalization (Registration to Template)**
    *   **Goal**: Align subject brain to the standard **MNI152** space.
    *   **Method**: 
        1.  **Affine**: 12-DOF global alignment.
        2.  **Non-Linear (Demons)**: Deformable registration for local warping.
    *   **Output**: Normalized T1w (`_space-MNI`), and crucial **Forward Transforms** (Affine Matrix + Warp Field).

### **Phase 2: Functional (Func)**

5.  **Dummy Scan Removal**
    *   **Goal**: Remove first few volumes until T1 magnetization stabilizes.
    *   **Method**: Trimming first $N$ volumes (default 0 or user-defined).

6.  **Slice Timing Correction (STC)**
    *   **Goal**: Correct for the time difference between slice acquisitions within a TR.
    *   **Method**: Native Cubic Spline Interpolation to the middle of the TR.
    *   **Input**: Raw EPI $\to$ **Output**: Time-aligned EPI.

7.  **Motion Correction (MoCo)**
    *   **Goal**: Correct head movement during the scan.
    *   **Method**: Rigid Body (6-DOF) alignment of every volume to a reference volume.
    *   **Key Detail**: Calculates **Motion Matrices** but does *not* resample the data yet.

8.  **Distortion Correction (SDC)** (Optional)
    *   **Goal**: Correct geometric distortions (stretching/squashing) in EPI images.
    *   **Method**: Creates a **Warp Field** based on fieldmaps or phase-difference estimate.

9.  **Coregistration**
    *   **Goal**: Align Functional (EPI) images to Anatomical (T1w) images.
    *   **Method**: Rigid Body (6-DOF) optimization maximizing Normalized Mutual Information (NMI).
    *   **Output**: **EPI-to-T1 Matrix**.

10. **Single-Shot Resampling (Spatial Transforms)**
    *   **Goal**: Apply all spatial changes in ONE step to preserve sharpness.
    *   **Transforms Chain**: 
        $$ \text{MNI} \leftarrow \text{T1} \leftarrow \text{EPI (Distorted)} \leftarrow \text{EPI (Original)} $$
        Combined Transform = [MoCo Matrices] + [SDC Warp] + [EPI-to-T1 Matrix] + [T1-to-MNI Warp].
    *   **Action**: Maps voxel data from the *Time-Corrected Input* directly to *MNI Space*.

### **Phase 3: Post-Processing & QC**

11. **Post-Processing**
    *   **Spatial Smoothing**: Applies a Gaussian Kernel (e.g., FWHM=6mm) to increase signal-to-noise ratio.
    *   **Temporal Filtering**: High-pass filtering (e.g., > 0.01Hz) to remove slow drift.
    *   **Scaling/Grand Mean Scaling**: Normalizes intensity across the run.
    *   **Confound Regression (CompCor)**: Extracts noise signals from WM/CSF masks (from Segmentation) to clean the data.

12. **Quality Control (QC)**
    *   Generates visual reports for verification:
        *   Registration Overlaps (Red contour on Gray scale).
        *   Motion plots (Parameters & Framewise Displacement).
        *   Segmentation contours.
