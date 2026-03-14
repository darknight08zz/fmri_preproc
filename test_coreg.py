import sys, os
_coreg_dir = 'D:/New folder (2)/fmri_preproc/coreg'
sys.path.insert(0, _coreg_dir)
os.chdir(_coreg_dir)
from coregister import run_coregistration

try:
    print("Running...")
    run_coregistration(
        ref_path         = 'D:/New folder (2)/fmri_preproc/converted/Patient_01/anat/series_301_MPRAGE.nii',
        source_path      = 'D:/New folder (2)/fmri_preproc/converted/Patient_01/func/meanseries_501_Resting_State_fMRI.nii',
        other_paths      = ['D:/New folder (2)/fmri_preproc/converted/Patient_01/func/arseries_501_Resting_State_fMRI.nii'],
        separation       = [4,2],
        hist_smooth_fwhm = [7,7],
        interp_order     = 4,
        wrap             = [0,0,0],
        mask             = False,
        prefix           = 'r',
        verbose          = True,
    )
    print("Done!")
except Exception as e:
    import traceback
    traceback.print_exc()
