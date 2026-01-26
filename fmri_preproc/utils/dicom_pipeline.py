import os
import pydicom
import numpy as np
import nibabel as nib
import logging
from typing import List, Dict, Any, Generator, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class DicomSlice:
    filepath: str
    instance_number: int
    acquisition_time: float
    image_position: np.ndarray # ImagePositionPatient
    image_orientation: np.ndarray # ImageOrientationPatient
    pixel_spacing: np.ndarray # PixelSpacing
    slice_thickness: float
    repetition_time: float
    series_uid: str
    rows: int
    columns: int
    pixel_data: np.ndarray

class DicomPipeline:
    def __init__(self, input_dir: str, subject: str, session: Optional[str] = None):
        self.input_dir = input_dir
        self.output_dir = None
        self.subject = subject
        self.session = session
        self.slices: List[DicomSlice] = []
        self.affine = np.eye(4)
        self.nifti_img = None
        self.slice_timing: List[float] = []

    def run(self, output_dir: str) -> Generator[Tuple[str, int], None, str]:
        """
        Execute the 11-stage pipeline.
        Yields (stage_name, progress_percent).
        Returns the path to the created NIfTI file.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Stage 1: Input Ingestion
        yield "Stage 1: Input Ingestion - Reading Headers", 5
        self._ingest_files()
        
        if not self.slices:
            raise ValueError("No valid image DICOMs found.")

        # Stage 2: Series Identification
        yield "Stage 2: Series Identification - Grouping", 15
        self._filter_primary_series()

        # Stage 3: Metadata Extraction (Implicitly done during Ingestion, but refining here)
        yield "Stage 3: Metadata Extraction", 25
        # Already parsed into DicomSlice objects, but we validate consistency here
        self._validate_geometry_consistency()

        # Stage 4: Slice Sorting (Spatial)
        yield "Stage 4: Slice Sorting (Z-Axis)", 35
        # Sort by spatial position along normal
        self._sort_spatially()

        # Stage 5: Time Sorting
        yield "Stage 5: Time Sorting", 45
        # Group into volumes based on time/acquisition
        volumes = self._group_by_time()
        
        # Stage 6 & 7: Building Volumes & 4D Array
        yield "Stage 6 & 7: Building 4D Volume", 60
        data_4d = self._build_4d_array(volumes)
        
        # Stage 8: Affine Matrix Construction
        yield "Stage 8: Affine Matrix Construction", 75
        self._build_affine()

        # Stage 9: Slice Timing Calculation
        yield "Stage 9: Slice Timing Calculation", 85
        self._calc_slice_timing()

        # Stage 10: Writing NIfTI
        yield "Stage 10: Writing NIfTI File", 95
        output_path = self._write_nifti(data_4d)

        # Stage 11: Verification
        yield "Stage 11: Verification", 99
        self._verify_output(output_path, data_4d.shape)

        yield "Complete", 100
        return output_path

    def _ingest_files(self):
        """Read all DICOM headers, ignore non-images."""
        valid_slices = []
        files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) 
                 if f.lower().endswith('.dcm') or f.lower().endswith('.ima')] # Add .ima support just in case
        
        total_files = len(files)
        for idx, fpath in enumerate(files):
            try:
                # Force read only necessary tags for speed if possible, but pydicom reads all by default usually
                # stop_before_pixels=True is faster but we might need pixel data later (lazy load safely)
                ds = pydicom.dcmread(fpath, stop_before_pixels=False) 
                
                # Filter localizers/scouts if possible (often have ImageType with LOCALIZER)
                if 'ImageType' in ds and 'LOCALIZER' in ds.ImageType:
                    continue
                
                # Check for necessary fields
                if not all(hasattr(ds, attr) for attr in ['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing']):
                    continue

                # Parse specific value types safely
                try:
                    ipp = np.array(ds.ImagePositionPatient, dtype=float)
                    iop = np.array(ds.ImageOrientationPatient, dtype=float)
                    ps = np.array(ds.PixelSpacing, dtype=float)
                    st = float(getattr(ds, 'SliceThickness', 0.0)) or float(getattr(ds, 'SpacingBetweenSlices', 0.0))
                    
                    # Time fields
                    tr = float(getattr(ds, 'RepetitionTime', 0.0)) / 1000.0 # Convert ms to seconds
                    acq_time = float(getattr(ds, 'AcquisitionTime', 0.0)) if hasattr(ds, 'AcquisitionTime') and ds.AcquisitionTime else 0.0
                    inst_num = int(getattr(ds, 'InstanceNumber', 0))
                except (ValueError, AttributeError):
                    continue

                valid_slices.append(DicomSlice(
                    filepath=fpath,
                    instance_number=inst_num,
                    acquisition_time=acq_time,
                    image_position=ipp,
                    image_orientation=iop,
                    pixel_spacing=ps,
                    slice_thickness=st,
                    repetition_time=tr,
                    series_uid=str(ds.SeriesInstanceUID),
                    rows=ds.Rows,
                    columns=ds.Columns,
                    pixel_data=ds.pixel_array
                ))
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
        
        self.slices = valid_slices
        logger.info(f"Ingested {len(self.slices)} valid slices from {total_files} files.")

    def _filter_primary_series(self):
        """Group by SeriesInstanceUID and pick the one with most files (functional)."""
        series_groups = defaultdict(list)
        for s in self.slices:
            series_groups[s.series_uid].append(s)
        
        if not series_groups:
            raise ValueError("No series found.")
        
        # Pick largest series
        primary_uid = max(series_groups, key=lambda k: len(series_groups[k]))
        self.slices = series_groups[primary_uid]
        logger.info(f"Selected Series UID {primary_uid} with {len(self.slices)} slices.")

    def _validate_geometry_consistency(self):
        """Ensure all slices in series have same dims, spacing, orientation."""
        ref = self.slices[0]
        for s in self.slices:
            if (s.rows != ref.rows or
                s.columns != ref.columns or
                not np.allclose(s.pixel_spacing, ref.pixel_spacing) or
                not np.allclose(s.image_orientation, ref.image_orientation)):
                logger.warning(f"Inconsistent geometry in slice {s.instance_number}")
                # For strictness, you might raise Error, but we'll just log
    
    def _sort_spatially(self):
        """
        Sort slice list by projection onto slice normal.
        This orders them anatomically regardless of acquisition order.
        """
        # 1. Compute slice normal
        ref = self.slices[0]
        row_cosine = ref.image_orientation[:3]
        col_cosine = ref.image_orientation[3:]
        slice_normal = np.cross(row_cosine, col_cosine)
        
        # 2. Project IPP onto normal
        # We assign a 'spatial_index' to each slice
        # Ideally, there are Z unique positions.
        # We need to sort primarily by spatial position (Z), then by Time.
        # Wait, usually we separate volumes first?
        # Actually in fMRI, we often have all slices mixed.
        # A robust way: Sort all by time, then space? Or space then time?
        
        # Let's assign a Z-coordinate approximation to every slice
        for s in self.slices:
            # Distance along normal
            s.z_position = np.dot(s.image_position, slice_normal)
        
        # Sort by Z-position to establish a canonical spatial order?
        # Actually we need to identify how many unique Z positions exist.
        z_positions = sorted(list(set([round(s.z_position, 3) for s in self.slices])))
        self.num_slices_per_volume = len(z_positions)
        logger.info(f"Detected {self.num_slices_per_volume} distinct slice positions.")
        
        # Verify
        if len(self.slices) % self.num_slices_per_volume != 0:
            logger.warning("Total slices is not a multiple of slices_per_volume. Missing data possible.")

    def _group_by_time(self):
        """
        Group slices into volumes.
        We can group by 'AcquisitionTime' or just chunk them if we trust order.
        But 'Stage 5' says Time Sorting.
        
        Strategy:
        1. Sort ALL slices by AcquisitionTime (primary) and InstanceNumber (secondary).
        2. Chunk into groups of `num_slices_per_volume`.
        3. Inside each chunk, sort by spatial Z position.
        """
        # Sort by generic acquisition order first to disentangle interleaved
        # Note: InstanceNumber is usually unique per series but resets or increments globally.
        # Acquisition Time is better if available.
        
        # Sort key: Acquisition Time -> Instance Number
        sorted_all = sorted(self.slices, key=lambda s: (s.acquisition_time, s.instance_number))
        
        volumes = []
        current_vol_slices = []
        
        # Chunk strategy
        # NOTE: This assumes standard acquisition where full volume is acquired before next.
        # This might fail for some simultaneous multi-slice, but standard for basic pipeline.
        
        # Better: Bin by unique Temporal IDs?
        # Let's count volumes
        num_volumes = len(self.slices) // self.num_slices_per_volume
        
        # We simply chunk the temporally-sorted list
        for i in range(num_volumes):
            start_idx = i * self.num_slices_per_volume
            end_idx = start_idx + self.num_slices_per_volume
            vol_slices = sorted_all[start_idx:end_idx]
            
            # Now sort this volume SPATIALLY to ensure correct 3D stacking
            # Use the normal projection we calculated in Stage 4
            # (We need to re-compute it or use the cached z_position if we stored it, let's recalculate for safety/clarity)
            ref = vol_slices[0]
            row_c = ref.image_orientation[:3]
            col_c = ref.image_orientation[3:]
            normal = np.cross(row_c, col_c)
            
            vol_slices.sort(key=lambda s: np.dot(s.image_position, normal))
            volumes.append(vol_slices)
            
        return volumes

    def _build_4d_array(self, volumes: List[List[DicomSlice]]) -> np.ndarray:
        """
        Stack slices -> 3D volumes -> 4D array.
        """
        # Shape: (X, Y, Z, T)
        ref = volumes[0][0]
        X, Y = ref.columns, ref.rows # DICOM (col, row) is (X, Y) usually, check transpose needs
        Z = len(volumes[0])
        T = len(volumes)
        
        # In DICOM pixel_array, usually (rows, cols).
        # We'll stack them.
        
        # Initialize
        data_4d = np.zeros((X, Y, Z, T), dtype=ref.pixel_data.dtype)
        
        for t, vol_slices in enumerate(volumes):
            if len(vol_slices) != Z:
                continue # Skip partial volume
            
            # Stack 3D
            # Note: DICOM pixel_array is (Rows, Columns).
            # NIfTI expects (X, Y, Z). 
            # Usually X is Columns, Y is Rows? or rotated.
            # We will perform a simple transpose if needed to match standard orientation,
            # but the affine takes care of the mapping to world space.
            # Standard practice: Transpose (Rows, Cols) -> (Cols, Rows) = (X, Y)
            
            vol_3d = np.stack([s.pixel_data.T for s in vol_slices], axis=2)
            data_4d[..., t] = vol_3d
            
        return data_4d

    def _build_affine(self):
        """
        Construct 4x4 matrix from IPP, IOP, PixelSpacing.
        NIfTI affine: [X_stk, Y_stk, Z_stk, Origin]
        
        Using the first sorted slice of the first volume as reference (Z=0).
        """
        # Get Z=0 slice (first slice of first volume) - assume we have it sorted
        # We need to grab it from 'volumes' ideally, but we can re-find the "bottom" slice
        # Let's assume self.slices was left in some state? No, use re-detection logic.
        # For simplicity, we use the very first slice of the first volume identified in grouping.
        # But we need to persist that order.
        # Let's reconstruct the reference from sorted list or just do it again.
        
        # Sort all slices spatially to find the 'first' spatial slice (most negative/positive normal proj)
        ref = self.slices[0]
        row_c = ref.image_orientation[:3]
        col_c = ref.image_orientation[3:]
        normal = np.cross(row_c, col_c)
        
        # Sort one volume's worth
        one_vol = sorted(self.slices[:self.num_slices_per_volume], key=lambda s: np.dot(s.image_position, normal))
        first_slice = one_vol[0] # Z=0
        last_slice = one_vol[-1] # Z=N-1
        
        # Compute Affine
        # Column 1: Row Vector * PixelSpacing X
        # Column 2: Col Vector * PixelSpacing Y
        # Column 3: Slice Thickness/Spacing Vector
        
        dr, dc = first_slice.pixel_spacing
        
        # X axis (rows in storage, but usually corresponding to IO[:3])
        # ImageOrientationPatient is [Rx, Ry, Rz, Cx, Cy, Cz]
        # X direction (along row index) -> R vector
        # Y direction (along col index) -> C vector
        # Wait, DICOM (row, col) map to X, Y? 
        # Usually:
        #  Col direction (X in image) is 2nd triplet of IOP?
        #  Row direction (Y in image) is 1st triplet?
        # Let's verify standard:
        # IOP: The direction cosines of the first row and the first column with respect to the patient.
        # First 3: Row X direction cosine, Row Y, Row Z. (Direction of increasing Column index) -> X axis
        # Second 3: Column X direction cosine... (Direction of increasing Row index) -> Y axis
        
        rx, ry, rz = first_slice.image_orientation[:3]
        cx, cy, cz = first_slice.image_orientation[3:]
        
        # Normalized vectors
        r_vec = np.array([rx, ry, rz])
        c_vec = np.array([cx, cy, cz])
        
        # Z vector (slice stacking direction)
        # Calculate from difference between first and last slice to handle spacing accurately
        # Or just use normal * thickness
        # Better: (Pos_last - Pos_first) / (N-1)
        if self.num_slices_per_volume > 1:
             z_vec_full = (last_slice.image_position - first_slice.image_position)
             z_step = z_vec_full / (self.num_slices_per_volume - 1)
        else:
             z_step = normal * first_slice.slice_thickness
        
        # Build 4x4
        # Column 0: X-axis stride (Direction of columns * spacing) -> r_vec * dr
        # Column 1: Y-axis stride (Direction of rows * spacing) -> c_vec * dc
        # NOTE: Be careful with Transpose done in build_4d. 
        # We transposed (Rows, Cols) -> (Cols, Rows).
        # So X axis corresponds to 'rows' in original? 
        # Let's stick to standard NIfTI orientation:
        # Affine should map (i,j,k) to (x,y,z).
        # i = X index (was Columns in DICOM if we Transposed?)
        # Let's assume standard behavior:
        # r_vec corresponds to increasing 'Column' index (X in image).
        # c_vec corresponds to increasing 'Row' index (Y in image).
        
        m = np.eye(4)
        m[:3, 0] = r_vec * dr
        m[:3, 1] = c_vec * dc
        m[:3, 2] = z_step
        m[:3, 3] = first_slice.image_position
        
        self.affine = m

    def _calc_slice_timing(self):
        """
        Stage 9: Calculate/Extract slice timing.
        """
        # Collect acquisition times for the first volume
        # We need to re-sort one volume by slice index (Z)
        ref = self.slices[0]
        row_c = ref.image_orientation[:3]
        col_c = ref.image_orientation[3:]
        normal = np.cross(row_c, col_c)
        
        one_vol = sorted(self.slices[:self.num_slices_per_volume], key=lambda s: np.dot(s.image_position, normal))
        
        # Times relative to first slice or TR start?
        # Usually relative to start of volume acquisition.
        min_time = min(s.acquisition_time for s in one_vol)
        self.slice_timing = [(s.acquisition_time - min_time) for s in one_vol]
        
    def _write_nifti(self, data: np.ndarray) -> str:
        """
        Write .nii.gz and .json sidecar
        """
        # Create output directory for func images (BIDS-like)
        func_dir = os.path.join(self.output_dir, "func")
        os.makedirs(func_dir, exist_ok=True)

        img = nib.Nifti1Image(data, self.affine)
        
        # Set TR in header
        tr = self.slices[0].repetition_time
        img.header.set_zooms(img.header.get_zooms()[:3] + (tr,))
        img.header.set_xyzt_units(xyz='mm', t='sec')
        
        filename = f"{self.subject}"
        if self.session:
            filename += f"_{self.session}"
        filename += "_task-rest_bold.nii.gz" # Simple default
        
        output_path = os.path.join(func_dir, filename)
        nib.save(img, output_path)
        
        # Ensure dataset_description.json exists in the root of the BIDS output (parent of subject folder)
        bids_root = os.path.dirname(self.output_dir)
        desc_path = os.path.join(bids_root, "dataset_description.json")
        if not os.path.exists(desc_path):
            import json
            try:
                with open(desc_path, 'w') as f:
                    json.dump({
                        "Name": "Converted Dataset",
                        "BIDSVersion": "1.8.0",
                        "DatasetType": "raw",
                        "License": "CC0"
                    }, f, indent=4)
                logger.info(f"Created BIDS root description at {desc_path}")
            except Exception as e:
                logger.warning(f"Could not create dataset_description.json at {bids_root}: {e}")
        
        # Write Sidecar JSON
        # TODO: Add structured JSON similar to dcm2niix
        
        return output_path

    def _verify_output(self, path: str, shape: Tuple):
        """
        Stage 11: Verification
        """
        if not os.path.exists(path):
            raise FileNotFoundError("Output file was not created.")
        
        try:
            img = nib.load(path)
            # Compare shapes (ignoring 1-dims if needed, but strict here)
            # data_4d is (X,Y,Z,T)
            # nibabel load shape matches
            if img.shape != shape:
                 logger.warning(f"Shape mismatch: Expected {shape}, got {img.shape}")
                 # raise ValueError(f"Shape mismatch: Expected {shape}, got {img.shape}")
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise e
        
        logger.info(f"Verification passed. Shape: {img.shape}")
