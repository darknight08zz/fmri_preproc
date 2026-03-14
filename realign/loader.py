import nibabel as nib
import numpy as np
from pathlib import Path


class VolumeLoader:
    """
    Loads a 4D NIfTI file and prepares SPM-style output filenames
    for the Realign (Estimate + Reslice) module.

    Fix applied vs original:
    - get_fdata(dtype='float32') added — avoids default float64 which
      doubles memory usage for large 4D fMRI files
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
        self.img = self.data = self.affine = self.header = None

    def load(self):
        """
        FIX: dtype='float32' passed to get_fdata().
        Original used float64 (default) — 2x memory for no benefit.
        float32 is sufficient precision for all fMRI preprocessing.
        """
        try:
            self.img = nib.load(self.file_path)

            if self.img.ndim != 4:
                raise ValueError(f"Expected 4D NIfTI, got: {self.img.shape}")

            self.data   = self.img.get_fdata(dtype='float32')  # FIX
            self.affine = self.img.affine
            self.header = self.img.header

            print(f"[LOADER] Loaded : {self.file_path.name}")
            print(f"[LOADER] Shape  : {self.data.shape}  (X, Y, Z, T)")

            return self.data, self.affine, self.header

        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI: {e}")

    def get_output_filenames(self):
        """
        SPM Realign output naming:
            r{name}.nii      resliced 4D
            mean{name}.nii   mean image
            rp_{stem}.txt    motion parameters
        """
        parent = self.file_path.parent
        stem   = self.file_path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]

        return {
            'resliced'     : parent / f"r{self.file_path.name}",
            'mean'         : parent / f"mean{self.file_path.name}",
            'motion_params': parent / f"rp_{stem}.txt"
        }