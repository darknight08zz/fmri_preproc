
import os
import nibabel as nib
import numpy as np
from pathlib import Path

class VolumeLoader:
    """
    Handles loading of 4D NIfTI files and extraction of metadata 
    for the Realign module.
    """
    
    def __init__(self, file_path):
        """
        Initialize the loader with the path to the 4D NIfTI file.
        
        Args:
            file_path (str): Path to the input 4D NIfTI file.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
            
        self.img = None
        self.data = None
        self.affine = None
        self.header = None
        
    def load(self):
        """
        Loads the NIfTI file and extracts data, affine, and header.
        
        Returns:
            tuple: (data, affine, header)
                - data (numpy.ndarray): 4D image data
                - affine (numpy.ndarray): 4x4 affine matrix
                - header (nibabel.nifti1.Nifti1Header): Image header
        """
        try:
            self.img = nib.load(self.file_path)
            
            # Ensure 4D
            if len(self.img.shape) != 4:
                raise ValueError(f"Input file must be 4D. Shape found: {self.img.shape}")
                
            self.data = self.img.get_fdata()
            self.affine = self.img.affine
            self.header = self.img.header
            
            return self.data, self.affine, self.header
            
        except Exception as e:
            raise RuntimeError(f"Error loading NIfTI file: {e}")

    def get_output_filenames(self):
        """
        Generates SPM-style output filenames based on the input filename.
        
        Returns:
            dict: Dictionary containing output paths:
                - 'resliced': Prefix 'r' (e.g., rsub-01_4D.nii)
                - 'mean': Prefix 'mean' (e.g., meansub-01_4D.nii)
                - 'motion_params': Prefix 'rp_' and .txt extension (e.g., rp_sub-01_4D.txt)
        """
        parent = self.file_path.parent
        stem = self.file_path.stem
        # Handle .nii.gz if necessary (pathlib stem handles .nii, but .nii.gz leaves .nii)
        if stem.endswith('.nii'):
            stem = stem[:-4]
            
        # Output filenames
        resliced_name = f"r{self.file_path.name}"
        mean_name = f"mean{self.file_path.name}"
        motion_params_name = f"rp_{stem}.txt"
        
        return {
            'resliced': parent / resliced_name,
            'mean': parent / mean_name,
            'motion_params': parent / motion_params_name
        }

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        loader = VolumeLoader(sys.argv[1])
        print(f"Loading {sys.argv[1]}...")
        data, affine, header = loader.load()
        print(f"Data shape: {data.shape}")
        outputs = loader.get_output_filenames()
        print("Output files would be:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
