
import nibabel as nib

class DummyScanRemoval:
    """
    Removes the first N volumes of a 4D fMRI series.
    """
    def run(self, input_path: str, output_path: str, dummy_scans: int = 4) -> bool:
        print(f"Removing {dummy_scans} dummy scans from {input_path}")
        try:
            img = nib.load(input_path)
            data = img.get_fdata()
            
            if len(data.shape) != 4:
                print("Error: Input is not a 4D file.")
                return False
                
            # Slice time dimension (4th dim)
            if data.shape[3] <= dummy_scans:
                 print("Error: Too few volumes.")
                 return False

            new_data = data[..., dummy_scans:]
            
            # Save
            new_img = nib.Nifti1Image(new_data, img.affine, img.header)
            nib.save(new_img, output_path)
            return True
        except Exception as e:
            print(f"Dummy scan removal failed: {e}")
            return False
