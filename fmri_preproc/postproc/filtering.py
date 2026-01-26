import numpy as np
import nibabel as nib
from scipy.signal import butter, filtfilt

class TemporalFiltering:
    """
    Temporal bandpass filtering.
    """
    def run(self, input_path: str, output_path: str, tr: float, highpass: float = 0.01, lowpass: float = 0.1) -> bool:
        print(f"Filtering {input_path} (Highpass: {highpass}Hz, Lowpass: {lowpass}Hz)")
        try:
            img = nib.load(input_path)
            data = img.get_fdata() # (X, Y, Z, T)
            
            if tr <= 0:
                print("Invalid TR.")
                return False
                
            # Nyquist
            nyq = 0.5 * (1.0/tr)
            
            # Normalize frequencies
            low = highpass / nyq
            high = lowpass / nyq
            
            # Check bounds
            if low <= 0 or high >= 1:
                print(f"Filter bounds invalid for TR={tr}: {low}-{high}")
                return False
            
            b, a = butter(2, [low, high], btype='band')
            
            # Filter along time axis (axis 3)
            # Use filtfilt for zero phase
            filtered_data = filtfilt(b, a, data, axis=-1)
            
            nib.save(nib.Nifti1Image(filtered_data, img.affine, img.header), output_path)
            return True
        except Exception as e:
            print(f"Filtering failed: {e}")
            return False
