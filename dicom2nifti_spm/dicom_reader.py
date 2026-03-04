import os
import pydicom
import numpy as np
from collections import defaultdict

def read_dicom_directory(directory):
    dicom_groups = defaultdict(list)
    print(f"Scanning {directory} for DICOM files...")
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=False, force=True)
                
                if "PixelData" not in ds or "SeriesInstanceUID" not in ds:
                    continue
                
                required_tags = [
                    "ImagePositionPatient", 
                    "ImageOrientationPatient", 
                    "PixelSpacing",
                    "SliceThickness"
                ]
                
                if not all(tag in ds for tag in required_tags):
                    continue
                
                dicom_groups[ds.SeriesInstanceUID].append(ds)
                
            except Exception:
                continue
                
    print(f"Found {len(dicom_groups)} series.")
    return dicom_groups

def sort_slices(slices):
    if not slices:
        return []
        
    iop = slices[0].ImageOrientationPatient
    row_cosines = np.array(iop[:3])
    col_cosines = np.array(iop[3:])
    
    slice_normal = np.cross(row_cosines, col_cosines)
    
    slices.sort(key=lambda s: np.dot(np.array(s.ImagePositionPatient), slice_normal))
    
    return slices
