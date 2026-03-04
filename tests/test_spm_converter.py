import unittest
import numpy as np
from pydicom.dataset import Dataset
from dicom2nifti_spm.geometry import compute_affine
from dicom2nifti_spm.dicom_reader import sort_slices

class TestSPMDicomConverter(unittest.TestCase):
    def setUp(self):
        self.slices = []
        for i in range(3):
            ds = Dataset()
            ds.ImagePositionPatient = [0.0, 0.0, float(i * 5.0)]
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.PixelSpacing = [2.0, 2.0]
            ds.SliceThickness = 5.0
            ds.InstanceNumber = i + 1
            self.slices.append(ds)
            
    def test_sorting(self):
        scrambled = [self.slices[2], self.slices[0], self.slices[1]]
        sorted_slices = sort_slices(scrambled)
        self.assertEqual(sorted_slices[0].InstanceNumber, 1)
        self.assertEqual(sorted_slices[2].InstanceNumber, 3)
        
    def test_affine_identity(self):
        affine = compute_affine(self.slices)
        expected = np.diag([-2.0, -2.0, 5.0, 1.0])
        np.testing.assert_array_almost_equal(affine, expected)
        
    def test_affine_rotated(self):
        ds = self.slices[0]
        ds.ImageOrientationPatient = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0] 
        slices = [ds] 
        
        affine = compute_affine(slices)
        
        expected = np.array([
            [ 0.,  2.,  0.,  0.],
            [-2.,  0.,  0.,  0.],
            [ 0.,  0.,  5.,  0.],
            [ 0.,  0.,  0.,  1.]
        ])
        
        np.testing.assert_array_almost_equal(affine, expected)

if __name__ == '__main__':
    unittest.main()
