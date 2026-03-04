
import sys
import os
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import nibabel as nib
    from stc.correction import SliceTimer
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_stc():
    """
    Simple test for STC logic.
    Creates a synthetic 4D dataset where signal is a sine wave.
    """
    print("Testing Slice Timing Correction...")
    
    try:
        # Create synthetic data
        # 10x10x5 volume, 20 time points
        nx, ny, nz, nt = 10, 10, 5, 20
        tr = 2.0
        
        # Create time axis
        time = np.arange(nt) * tr
        
        # Create data: Sine wave with phase shift depending on Z
        data = np.zeros((nx, ny, nz, nt))
        freq = 0.1 # 0.1 Hz
        
        slice_offsets = np.linspace(0, tr - tr/nz, nz)
        
        for z in range(nz):
            # Signal in this slice is shifted
            shift = slice_offsets[z]
            # data[t] = sin(2*pi*freq * (time + shift))
            data[:, :, z, :] = np.sin(2 * np.pi * freq * (time + shift)).reshape(1, 1, nt)
            
        print("Data created. Running STC...")
        
        timer = SliceTimer(data, tr, nz)
        corrected = timer.correct(slice_order_type='ascending', ref_slice=0)
        
        # Check error
        target_signal = data[0, 0, 0, :]
        
        max_error = 0
        for z in range(nz):
            corrected_signal = corrected[0, 0, z, :]
            diff = np.max(np.abs(corrected_signal - target_signal))
            max_error = max(max_error, diff)
            print(f"Slice {z}: Max error = {diff:.6f}")
            
        print(f"Total Max Error: {max_error:.6f}")
        if max_error < 1e-4:
            print("PASS: Error is negligible.")
        else:
            print("FAIL: Error is too high.")
            
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_stc()
