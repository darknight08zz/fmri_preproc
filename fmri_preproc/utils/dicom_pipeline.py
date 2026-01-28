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
    series_description: str
    rows: int
    columns: int
    pixel_data: np.ndarray
    # Extended metadata for BIDS
    flip_angle: Optional[float] = None
    manufacturer: Optional[str] = None
    manufacturers_model_name: Optional[str] = None
    magnetic_field_strength: Optional[float] = None

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
        self.num_slices_per_volume = 0

    def run(self, output_dir: str) -> Generator[Tuple[str, int], None, List[str]]:
        """
        Execute the pipeline for ALL detected series.
        Yields (stage_name, progress_percent).
        Returns list of paths to created NIfTI files.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Stage 1: Input Ingestion
        yield "Stage 1: Input Ingestion - Reading Headers", 5
        self._ingest_files()
        
        if not self.slices:
            raise ValueError("No valid image DICOMs found.")

        # Group by series
        series_map = defaultdict(list)
        for s in self.slices:
            series_map[s.series_uid].append(s)
            
        generated_files = []
        total_series = len(series_map)
        
        # Process each series
        for idx, (uid, slices) in enumerate(series_map.items()):
            try:
                # Classify
                modality, suffix = self._classify_series(slices)
                if modality == 'ignore':
                    logger.info(f"Ignoring series {uid} ({slices[0].series_description})")
                    continue
                
                series_name = f"{modality.upper()} ({slices[0].series_description})"
                base_progress = 10 + int((idx / total_series) * 80)
                
                yield f"Processing {series_name}", base_progress
                
                # Context switch for this series
                self.slices = slices
                
                # Processing Steps
                self._validate_geometry_consistency()
                self._sort_spatially()
                volumes = self._group_by_time()
                data_4d = self._build_4d_array(volumes)
                self._build_affine()
                self._calc_slice_timing()
                
                # Write
                out_path = self._write_nifti(data_4d, modality, suffix)
                generated_files.append(out_path)
                
            except Exception as e:
                logger.error(f"Failed to process series {uid}: {e}")
                continue

        yield "Complete", 100
        return generated_files

    def _classify_series(self, slices: List[DicomSlice]) -> Tuple[str, str]:
        """Determine if series is anat, func, or ignore."""
        desc = slices[0].series_description.lower()
        num_slices = len(slices)
        
        # Simple heuristics
        if any(x in desc for x in ['t1', 'mprage', 'anat']):
            return 'anat', '_T1w'
        elif any(x in desc for x in ['bold', 'fmri', 'rest', 'task']):
            return 'func', '_task-rest_bold'
        
        # Fallback based on volume count logic could go here, 
        # but for now let's be conservative and default to func if it looks like a timeseries?
        # Or just ignore unknown.
        # Let's check typical image types
        if num_slices > 50: # Arbitrary threshold, assume it's something useful
             # If it hasn't matched 't1', maybe it's functional?
             return 'func', '_task-rest_bold'
             
        return 'ignore', ''

    def _ingest_files(self):
        """Read all DICOM headers, ignore non-images."""
        valid_slices = []
        files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) 
                 if f.lower().endswith('.dcm') or f.lower().endswith('.ima')] 
        
        total_files = len(files)
        for idx, fpath in enumerate(files):
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=False) 
                
                if 'ImageType' in ds and 'LOCALIZER' in ds.ImageType:
                    continue
                
                if not all(hasattr(ds, attr) for attr in ['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing']):
                    continue

                try:
                    ipp = np.array(ds.ImagePositionPatient, dtype=float)
                    iop = np.array(ds.ImageOrientationPatient, dtype=float)
                    ps = np.array(ds.PixelSpacing, dtype=float)
                    st = float(getattr(ds, 'SliceThickness', 0.0)) or float(getattr(ds, 'SpacingBetweenSlices', 0.0))
                    
                    tr = float(getattr(ds, 'RepetitionTime', 0.0)) / 1000.0 # Convert ms to seconds
                    acq_time = float(getattr(ds, 'AcquisitionTime', 0.0)) if hasattr(ds, 'AcquisitionTime') and ds.AcquisitionTime else 0.0
                    inst_num = int(getattr(ds, 'InstanceNumber', 0))
                    
                    # Extended metadata
                    fa = float(getattr(ds, 'FlipAngle', 0.0)) if hasattr(ds, 'FlipAngle') and ds.FlipAngle else None
                    manuf = str(getattr(ds, 'Manufacturer', '')) if hasattr(ds, 'Manufacturer') else None
                    model = str(getattr(ds, 'ManufacturerModelName', '')) if hasattr(ds, 'ManufacturerModelName') else None
                    mfs = float(getattr(ds, 'MagneticFieldStrength', 0.0)) if hasattr(ds, 'MagneticFieldStrength') and ds.MagneticFieldStrength else None
                    desc = str(getattr(ds, 'SeriesDescription', 'Unknown'))

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
                    series_description=desc,
                    rows=ds.Rows,
                    columns=ds.Columns,
                    pixel_data=ds.pixel_array,
                    flip_angle=fa,
                    manufacturer=manuf,
                    manufacturers_model_name=model,
                    magnetic_field_strength=mfs
                ))
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
        
        self.slices = valid_slices
        logger.info(f"Ingested {len(self.slices)} valid slices from {total_files} files.")

    def _validate_geometry_consistency(self):
        """Ensure all slices in series have same dims, spacing, orientation."""
        ref = self.slices[0]
        for s in self.slices:
            if (s.rows != ref.rows or
                s.columns != ref.columns or
                not np.allclose(s.pixel_spacing, ref.pixel_spacing) or
                not np.allclose(s.image_orientation, ref.image_orientation)):
                logger.warning(f"Inconsistent geometry in slice {s.instance_number}")
    
    def _sort_spatially(self):
        """Sort slice list by projection onto slice normal."""
        ref = self.slices[0]
        row_cosine = ref.image_orientation[:3]
        col_cosine = ref.image_orientation[3:]
        slice_normal = np.cross(row_cosine, col_cosine)
        
        for s in self.slices:
            s.z_position = np.dot(s.image_position, slice_normal)
        
        z_positions = sorted(list(set([round(s.z_position, 3) for s in self.slices])))
        self.num_slices_per_volume = len(z_positions)
        logger.info(f"Detected {self.num_slices_per_volume} distinct slice positions.")
        
        if self.num_slices_per_volume == 0:
             raise ValueError("Could not determine number of slices per volume.")

        if len(self.slices) % self.num_slices_per_volume != 0:
            logger.warning("Total slices is not a multiple of slices_per_volume. Missing data possible.")

    def _group_by_time(self):
        """Group slices into volumes."""
        sorted_all = sorted(self.slices, key=lambda s: (s.acquisition_time, s.instance_number))
        
        volumes = []
        num_volumes = len(self.slices) // self.num_slices_per_volume
        
        for i in range(num_volumes):
            start_idx = i * self.num_slices_per_volume
            end_idx = start_idx + self.num_slices_per_volume
            vol_slices = sorted_all[start_idx:end_idx]
            
            ref = vol_slices[0]
            row_c = ref.image_orientation[:3]
            col_c = ref.image_orientation[3:]
            normal = np.cross(row_c, col_c)
            
            vol_slices.sort(key=lambda s: np.dot(s.image_position, normal))
            volumes.append(vol_slices)
            
        return volumes

    def _build_4d_array(self, volumes: List[List[DicomSlice]]) -> np.ndarray:
        """Stack slices -> 3D volumes -> 4D array."""
        if not volumes:
            raise ValueError("No volumes constructed.")
            
        ref = volumes[0][0]
        X, Y = ref.columns, ref.rows 
        Z = len(volumes[0])
        T = len(volumes)
        
        data_4d = np.zeros((X, Y, Z, T), dtype=ref.pixel_data.dtype)
        
        for t, vol_slices in enumerate(volumes):
            if len(vol_slices) != Z:
                continue 
            
            vol_3d = np.stack([s.pixel_data.T for s in vol_slices], axis=2)
            data_4d[..., t] = vol_3d
            
        return data_4d

    def _build_affine(self):
        """Construct 4x4 NIfTI affine matrix."""
        ref = self.slices[0]
        row_c = ref.image_orientation[:3]
        col_c = ref.image_orientation[3:]
        normal = np.cross(row_c, col_c)
        
        one_vol = sorted(self.slices[:self.num_slices_per_volume], key=lambda s: np.dot(s.image_position, normal))
        first_slice = one_vol[0] 
        last_slice = one_vol[-1] 
        
        dr, dc = first_slice.pixel_spacing
        r_vec = np.array(first_slice.image_orientation[:3])
        c_vec = np.array(first_slice.image_orientation[3:])
        
        if self.num_slices_per_volume > 1:
             z_vec_full = (last_slice.image_position - first_slice.image_position)
             z_step = z_vec_full / (self.num_slices_per_volume - 1)
        else:
             z_step = normal * first_slice.slice_thickness
        
        m = np.eye(4)
        m[:3, 0] = r_vec * dr
        m[:3, 1] = c_vec * dc
        m[:3, 2] = z_step
        m[:3, 3] = first_slice.image_position
        
        self.affine = m

    def _calc_slice_timing(self):
        """Calculate slice timing relative to acquisition start."""
        ref = self.slices[0]
        row_c = ref.image_orientation[:3]
        col_c = ref.image_orientation[3:]
        normal = np.cross(row_c, col_c)
        
        one_vol = sorted(self.slices[:self.num_slices_per_volume], key=lambda s: np.dot(s.image_position, normal))
        
        min_time = min(s.acquisition_time for s in one_vol)
        self.slice_timing = [(s.acquisition_time - min_time) for s in one_vol]
        
    def _write_nifti(self, data: np.ndarray, modality: str, suffix: str) -> str:
        """Write .nii.gz and .json sidecar, ensuring BIDS compliance in ROOT."""
        # Create output directory for modality
        mod_dir = os.path.join(self.output_dir, modality)
        os.makedirs(mod_dir, exist_ok=True)
        
        # If anatomical (3D), squeeze time dimension if it exists and is 1
        if modality == 'anat' and data.ndim == 4 and data.shape[3] == 1:
            data = data.squeeze(axis=3)

        img = nib.Nifti1Image(data, self.affine)
        
        # Set TR in header
        if self.slices:
            tr = self.slices[0].repetition_time
            img.header.set_zooms(img.header.get_zooms()[:3] + (tr,))
            img.header.set_xyzt_units(xyz='mm', t='sec')
        
        filename = f"{self.subject}"
        if self.session:
            filename += f"_{self.session}"
        filename += suffix
        
        output_path = os.path.join(mod_dir, filename)
        nib.save(img, output_path)
        
        # CRITICAL FIX: Ensure dataset_description.json exists in the BIDS root (parent of subject dir)
        # self.output_dir is .../sub-XX
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

        # Create JSON Sidecar for the NIfTI
        json_filename = filename.replace('.nii.gz', '.json')
        json_path = os.path.join(mod_dir, json_filename)
        
        try:
            # Gather metadata from the first slice of the volume
            ref_slice = self.slices[0]
            
            sidecar_data = {
                "Modality": modality,
                "RepetitionTime": ref_slice.repetition_time,
                "SliceTiming": self.slice_timing if self.slice_timing and modality == 'func' else None,
                "ImageOrientationPatientDICOM": ref_slice.image_orientation.tolist() if hasattr(ref_slice.image_orientation, 'tolist') else ref_slice.image_orientation,
                "SliceThickness": ref_slice.slice_thickness,
            }
            if modality == 'func':
                 sidecar_data["TaskName"] = "rest"
            
            # Add optional fields if available
            if ref_slice.flip_angle is not None:
                sidecar_data["FlipAngle"] = ref_slice.flip_angle
            if ref_slice.manufacturer:
                sidecar_data["Manufacturer"] = ref_slice.manufacturer
            if ref_slice.manufacturers_model_name:
                sidecar_data["ManufacturersModelName"] = ref_slice.manufacturers_model_name
            if ref_slice.magnetic_field_strength:
                sidecar_data["MagneticFieldStrength"] = ref_slice.magnetic_field_strength

            import json
            with open(json_path, 'w') as jf:
                json.dump(sidecar_data, jf, indent=4)
            logger.info(f"Created BIDS sidecar at {json_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create JSON sidecar: {e}")

        # Ensure participants.tsv exists
        participants_path = os.path.join(bids_root, "participants.tsv")
        if not os.path.exists(participants_path):
            try:
                with open(participants_path, 'w') as pf:
                    pf.write("participant_id\tsex\tage\n")
                    pf.write(f"{self.subject}\tn/a\tn/a\n") # Initialize with current subject
                logger.info(f"Created participants.tsv at {participants_path}")
            except Exception as e:
                logger.warning(f"Failed to create participants.tsv: {e}")
        else:
            # Check if subject is already in participants.tsv, if not append
            try:
                with open(participants_path, 'r') as pf:
                    lines = pf.readlines()
                
                existing_ids = [line.split('\t')[0].strip() for line in lines]
                if self.subject not in existing_ids:
                    with open(participants_path, 'a') as pf:
                        pf.write(f"{self.subject}\tn/a\tn/a\n")
                    logger.info(f"Added {self.subject} to participants.tsv")
            except Exception as e:
                 logger.warning(f"Failed to update participants.tsv: {e}")
        
        return output_path

    def _verify_output(self, path: str, shape: Tuple):
        """Verify output file exists and has correct shape."""
        if not os.path.exists(path):
            raise FileNotFoundError("Output file was not created.")
        
        try:
            img = nib.load(path)
            if img.shape != shape:
                 logger.warning(f"Shape mismatch: Expected {shape}, got {img.shape}")
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise e
