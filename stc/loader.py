import nibabel as nib
from pathlib import Path

class VolumeLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self):
        img = nib.load(str(self.file_path))
        data = img.get_fdata(dtype="float32")

        if data.ndim != 4:
            raise ValueError("Input NIfTI must be 4D")

        affine = img.affine
        header = img.header

        tr = None
        try:
            tr = header.get_zooms()[3]
        except Exception:
            pass

        return data, affine, header, tr

    def output_path(self):
        return self.file_path.parent / f"a{self.file_path.name}"
