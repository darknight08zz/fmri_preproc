import nibabel as nib
from pathlib import Path


class VolumeLoader:
    """
    Loads a 4D NIfTI file and extracts:
        - data array (float32)
        - affine matrix
        - header
        - TR (repetition time in seconds)

    Fixes applied vs original:
    - TR=0 or missing now raises a clear error instead of silently returning None
    - TR validity check added (must be positive)
    - Informative error messages added throughout
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"NIfTI file not found: {self.file_path}"
            )

        # Only accept .nii or .nii.gz
        suffixes = ''.join(self.file_path.suffixes)
        if suffixes not in ('.nii', '.nii.gz'):
            raise ValueError(
                f"Expected .nii or .nii.gz file, got: {suffixes}"
            )

    def load(self):
        """
        Load NIfTI file.

        Returns:
            data   (np.ndarray) : shape (X, Y, Z, T), float32
            affine (np.ndarray) : 4x4 voxel-to-world matrix
            header             : NIfTI header object
            tr     (float)     : repetition time in seconds
        """

        img    = nib.load(str(self.file_path))
        data   = img.get_fdata(dtype='float32')
        affine = img.affine
        header = img.header

        # -- Validate dimensions --
        if data.ndim != 4:
            raise ValueError(
                f"Expected 4D NIfTI (X,Y,Z,T), got shape: {data.shape}"
            )

        # -- Extract TR --
        # FIX: original code silently returned None on failure.
        # Now we validate TR and raise a clear error if missing/zero.
        tr = self._extract_tr(header)

        print(f"[LOADER] File loaded  : {self.file_path.name}")
        print(f"[LOADER] Shape        : {data.shape}  (X, Y, Z, T)")
        print(f"[LOADER] nSlices      : {data.shape[2]}")
        print(f"[LOADER] nVolumes     : {data.shape[3]}")
        print(f"[LOADER] TR           : {tr:.4f} s")
        print(f"[LOADER] Affine       :\n{affine}")

        return data, affine, header, tr

    def _extract_tr(self, header):
        """
        Extract TR from NIfTI header pixdim[4] (the time dimension).

        NIfTI spec: pixdim[4] = voxel size along 4th dimension.
        For fMRI, this is TR in seconds (if xyzt_units time flag is set).

        FIX: Added check for TR=0 or negative, which indicates
        the NIfTI header was not properly set during conversion.
        In that case, raise a clear error prompting the user to
        provide TR manually via the JSON sidecar.
        """
        tr = None

        try:
            zooms = header.get_zooms()
            if len(zooms) >= 4:
                tr = float(zooms[3])
        except Exception as e:
            raise RuntimeError(
                f"Failed to read TR from NIfTI header: {e}\n"
                f"Please provide TR manually from your _bold.json file."
            )

        # FIX: TR=0 means header was not set (common in some DICOM converters)
        if tr is None or tr <= 0:
            raise ValueError(
                f"TR extracted from NIfTI header is invalid: {tr}\n"
                f"This usually means the DICOM-to-NIfTI conversion did not\n"
                f"write TR into pixdim[4]. Please provide TR manually:\n"
                f"  tr = your_json['RepetitionTime']  # from _bold.json"
            )

        return tr

    def output_path(self):
        """
        Returns output path with 'a' prefix (SPM STC convention).
        e.g. rfunc_4D.nii → arfunc_4D.nii
        """
        return self.file_path.parent / f"a{self.file_path.name}"