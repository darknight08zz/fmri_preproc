import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmri_preproc.qc.nodes import QCNode
from fmri_preproc.func.coregistration import Coregistration
from fmri_preproc.anat.segmentation import Segmentation
from fmri_preproc.anat.normalization import Normalization

def create_dummy_nifti(path, shape=(20, 20, 20), is_4d=False):
    data = np.random.rand(*shape) if not is_4d else np.random.rand(*shape, 5)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)
    return path

def test_qc_features():
    os.makedirs("qc_test_out", exist_ok=True)
    
    # 1. Test Motion Metrics
    print("Testing Motion Metrics...")
    par_file = "qc_test_out/motion.par"
    # Create valid 6-col param file
    params = np.random.rand(10, 6) # 10 vols
    np.savetxt(par_file, params)
    
    qc = QCNode()
    metrics = qc._calculate_motion_metrics(par_file)
    print("Motion Metrics:", metrics)
    assert 'Mean FD (mm)' in metrics
    assert 'QC Decision' in metrics
    assert metrics['QC Decision'] in ["PASS", "WARN", "FAIL"]
    
    # Test Plot
    qc._generate_motion_plot(par_file, "qc_test_out/motion_plot.png", metrics)
    assert os.path.exists("qc_test_out/motion_plot.png")
    
    # 2. Test Coreg Plot
    print("Testing Coreg Plot...")
    ref = create_dummy_nifti("qc_test_out/ref.nii.gz", shape=(20,30,20))
    src = create_dummy_nifti("qc_test_out/src.nii.gz", shape=(20,30,20))
    
    Coregistration()._generate_qc_plot(ref, src, "qc_test_out/coreg_qc.png")
    assert os.path.exists("qc_test_out/coreg_qc.png")
    
    # 3. Test Seg Plot
    print("Testing Seg Plot...")
    t1 = create_dummy_nifti("qc_test_out/t1.nii.gz", shape=(30,30,30))
    # Maps
    maps = [
        np.random.rand(30,30,30) for _ in range(3)
    ]
    # Need to save/load? The function takes maps as DATA list?
    # No, _generate_qc_plot signature in Segmentation is: (t1_path, maps, sorted_indices, out_path)
    # in NEW impl, `maps` are passed as list of arrays? 
    # Let's check my replacement content for Segmentation.
    # Step 80: csf_vol = maps[0], so maps is list of volumes (numpy arrays).
    # Wait, in the Node (Step 28), tissue_maps is list of STRINGS (paths).
    # In `segmentation.py` run(), it calls `_generate_qc_plot(..., maps, ...)`
    # In `segmentation.py` line 106: `maps = [np.zeros_like(data)...]` -> These are arrays.
    # So `_generate_qc_plot` expects ARRAYS.
    # My verification should pass arrays.
    
    seg = Segmentation()
    seg._generate_qc_plot(t1, maps, [0,1,2], "qc_test_out/seg_qc.png")
    assert os.path.exists("qc_test_out/seg_qc.png")
    
    # 4. Test Norm Plot
    print("Testing Norm Plot...")
    tpl = create_dummy_nifti("qc_test_out/tpl.nii.gz", shape=(40,40,40))
    warped = create_dummy_nifti("qc_test_out/warped.nii.gz", shape=(40,40,40))
    
    Normalization()._generate_qc_plot(tpl, warped, "qc_test_out/norm_qc.png")
    assert os.path.exists("qc_test_out/norm_qc.png")

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_qc_features()
