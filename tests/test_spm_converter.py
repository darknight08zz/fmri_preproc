import unittest
import numpy as np
from pydicom.dataset import Dataset
from dicom2nifti_spm.geometry import compute_affine
from dicom2nifti_spm.dicom_reader import sort_slices


def _make_slice(position, iop, pixel_spacing, slice_thickness, instance_number):
    """Helper: build a minimal mock DICOM Dataset."""
    ds = Dataset()
    ds.ImagePositionPatient    = [float(x) for x in position]
    ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.PixelSpacing            = [float(x) for x in pixel_spacing]
    ds.SliceThickness          = float(slice_thickness)
    ds.InstanceNumber          = instance_number
    return ds


# Standard axial IOP (identity-like orientation)
AXIAL_IOP = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class TestSortSlices(unittest.TestCase):
    """
    Tests for sort_slices() — sorts by spatial position
    projected onto the slice normal, NOT by InstanceNumber.
    """

    def setUp(self):
        # 3 axial slices at Z = 0, 5, 10 mm
        self.slices = [
            _make_slice([0, 0, 0],  AXIAL_IOP, [2, 2], 5, 1),
            _make_slice([0, 0, 5],  AXIAL_IOP, [2, 2], 5, 2),
            _make_slice([0, 0, 10], AXIAL_IOP, [2, 2], 5, 3),
        ]

    def test_sorting_restores_spatial_order(self):
        """
        Scrambled slices are restored to ascending Z-position order.
        sort_slices() uses ImagePositionPatient projection onto the
        slice normal — not InstanceNumber.
        """
        scrambled   = [self.slices[2], self.slices[0], self.slices[1]]
        sorted_s    = sort_slices(scrambled)

        positions = [s.ImagePositionPatient[2] for s in sorted_s]
        self.assertEqual(positions, [0.0, 5.0, 10.0])

    def test_sorting_spatial_beats_instance_number(self):
        """
        FIX for original test: original test verified InstanceNumber order,
        but sort_slices sorts by SPATIAL position, not InstanceNumber.

        This test uses slices where spatial Z-order is OPPOSITE to
        InstanceNumber order — confirming the function sorts spatially.

        Slice A: Z=10, InstanceNumber=1  (high Z, low instance)
        Slice B: Z=5,  InstanceNumber=2
        Slice C: Z=0,  InstanceNumber=3  (low Z, high instance)

        sort_slices must return [C, B, A] (ascending Z),
        NOT [A, B, C] (ascending InstanceNumber).
        """
        slice_A = _make_slice([0, 0, 10], AXIAL_IOP, [2, 2], 5, instance_number=1)
        slice_B = _make_slice([0, 0,  5], AXIAL_IOP, [2, 2], 5, instance_number=2)
        slice_C = _make_slice([0, 0,  0], AXIAL_IOP, [2, 2], 5, instance_number=3)

        sorted_s = sort_slices([slice_A, slice_B, slice_C])

        # Spatial Z should be ascending: 0, 5, 10
        positions = [s.ImagePositionPatient[2] for s in sorted_s]
        self.assertEqual(positions, [0.0, 5.0, 10.0])

        # InstanceNumbers in result will be [3, 2, 1] — opposite to input order
        instance_numbers = [s.InstanceNumber for s in sorted_s]
        self.assertEqual(instance_numbers, [3, 2, 1])

    def test_sorting_single_slice(self):
        """Single slice should be returned unchanged."""
        result = sort_slices([self.slices[0]])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].ImagePositionPatient[2], 0.0)

    def test_sorting_empty(self):
        """Empty input should return empty list."""
        result = sort_slices([])
        self.assertEqual(result, [])


class TestComputeAffine(unittest.TestCase):
    """
    Tests for compute_affine() — builds 4×4 RAS affine from DICOM geometry.
    """

    def test_affine_axial_zero_origin(self):
        """
        Axial orientation, 2mm pixels, 5mm slice gap, origin at [0,0,0].

        IOP = [1,0,0, 0,1,0]
        r_vector=[1,0,0], c_vector=[0,1,0]
        slice_normal = cross([1,0,0],[0,1,0]) = [0,0,1]

        affine_lps:
          col_step  = r_vector * dx = [2,0,0]
          row_step  = c_vector * dy = [0,2,0]
          slice_step= normal  * dz = [0,0,5]
          origin    = [0,0,0]

        After LPS→RAS (flip X,Y rows):
          [[-2, 0, 0, 0],
           [ 0,-2, 0, 0],
           [ 0, 0, 5, 0],
           [ 0, 0, 0, 1]]
        """
        slices = [
            _make_slice([0, 0, 0],  AXIAL_IOP, [2, 2], 5, 1),
            _make_slice([0, 0, 5],  AXIAL_IOP, [2, 2], 5, 2),
            _make_slice([0, 0, 10], AXIAL_IOP, [2, 2], 5, 3),
        ]
        affine = compute_affine(slices)

        expected = np.array([
            [-2.,  0.,  0.,  0.],
            [ 0., -2.,  0.,  0.],
            [ 0.,  0.,  5.,  0.],
            [ 0.,  0.,  0.,  1.]
        ])
        np.testing.assert_array_almost_equal(affine, expected)

    def test_affine_axial_realistic_origin(self):
        """
        FIX for original test_affine_identity:

        Original test used ImagePositionPatient=[0,0,0] and expected
        np.diag([-2,-2,5,1]) which zeroes the origin column.

        This ONLY passes because origin=[0,0,0] makes the origin term
        vanish. For any real scanner data (e.g. ADNI origin [-90,-126,-72]),
        the origin column is non-zero and the diag test would FAIL.

        This test uses a realistic scanner origin and verifies the
        origin column is correctly transformed (LPS X,Y flipped to RAS).

        Origin LPS = [-90, -126, -72]
        Origin RAS = [+90, +126, -72]  (X and Y flipped)
        """
        slices = [
            _make_slice([-90, -126, -72], AXIAL_IOP, [2, 2], 5, 1),
            _make_slice([-90, -126, -67], AXIAL_IOP, [2, 2], 5, 2),
            _make_slice([-90, -126, -62], AXIAL_IOP, [2, 2], 5, 3),
        ]
        affine = compute_affine(slices)

        expected = np.array([
            [-2.,  0.,  0.,  90.],
            [ 0., -2.,  0., 126.],
            [ 0.,  0.,  5., -72.],
            [ 0.,  0.,  0.,   1.]
        ])
        np.testing.assert_array_almost_equal(affine, expected)

    def test_affine_dz_from_slice_positions(self):
        """
        dz must be computed from (last_pos - first_pos) / (n-1),
        NOT from SliceThickness tag.

        SliceThickness = 5mm but actual slice gap = 6mm.
        Affine must reflect the real gap (6mm), not the tag (5mm).
        """
        slices = [
            _make_slice([0, 0,  0], AXIAL_IOP, [2, 2], slice_thickness=5, instance_number=1),
            _make_slice([0, 0,  6], AXIAL_IOP, [2, 2], slice_thickness=5, instance_number=2),
            _make_slice([0, 0, 12], AXIAL_IOP, [2, 2], slice_thickness=5, instance_number=3),
        ]
        affine = compute_affine(slices)

        # Z step must be 6mm (actual gap), not 5mm (SliceThickness tag)
        self.assertAlmostEqual(affine[2, 2], 6.0)

    def test_affine_single_slice_uses_thickness(self):
        """
        For a single slice, dz cannot be computed from positions.
        Must fall back to SliceThickness tag.
        """
        slices = [
            _make_slice([0, 0, 0], AXIAL_IOP, [2, 2], slice_thickness=4, instance_number=1),
        ]
        affine = compute_affine(slices)
        self.assertAlmostEqual(affine[2, 2], 4.0)

    def test_affine_rotated_iop(self):
        """
        90° rotated IOP: row direction = Y, column direction = -X.

        IOP = [0,1,0, -1,0,0]
        r_vector = [0,1,0]
        c_vector = [-1,0,0]
        slice_normal = cross([0,1,0],[-1,0,0]) = [0,0,1]

        affine_lps:
          col_step   = r_vector * dx = [0,2,0]
          row_step   = c_vector * dy = [-2,0,0]
          slice_step = [0,0,5]

        After LPS→RAS:
          col 0: flip X,Y → [0,-2,0]   wait — let's be precise:
          lps_to_ras flips rows 0 and 1:
            affine_lps col0 = [0,2,0]  → ras col0 = [-0,-2,0] = [0,-2,0]
            affine_lps col1 = [-2,0,0] → ras col1 = [2, 0, 0]

        Expected RAS affine:
          [[ 0,  2,  0,  0],
           [-2,  0,  0,  0],
           [ 0,  0,  5,  0],
           [ 0,  0,  0,  1]]
        """
        rotated_iop = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
        slices = [
            _make_slice([0, 0, 0], rotated_iop, [2, 2], slice_thickness=5, instance_number=1),
        ]
        affine = compute_affine(slices)

        expected = np.array([
            [ 0.,  2.,  0.,  0.],
            [-2.,  0.,  0.,  0.],
            [ 0.,  0.,  5.,  0.],
            [ 0.,  0.,  0.,  1.]
        ])
        np.testing.assert_array_almost_equal(affine, expected)

    def test_affine_lps_to_ras_flip(self):
        """
        LPS→RAS conversion must flip X and Y components of origin,
        and flip X and Y components of all direction vectors.
        """
        slices = [
            _make_slice([10, 20, 30], AXIAL_IOP, [2, 2], 5, 1),
            _make_slice([10, 20, 35], AXIAL_IOP, [2, 2], 5, 2),
        ]
        affine = compute_affine(slices)

        # X origin: LPS x=10 → RAS x=-10  (flipped)
        self.assertAlmostEqual(affine[0, 3], -10.0)
        # Y origin: LPS y=20 → RAS y=-20  (flipped)
        self.assertAlmostEqual(affine[1, 3], -20.0)
        # Z origin: LPS z=30 → RAS z=+30  (unchanged)
        self.assertAlmostEqual(affine[2, 3],  30.0)


class TestComputeAffineEdgeCases(unittest.TestCase):
    """Edge cases and robustness checks."""

    def test_affine_empty_raises(self):
        """compute_affine must raise ValueError on empty input."""
        with self.assertRaises(ValueError):
            compute_affine([])

    def test_affine_non_unit_iop_normalized(self):
        """
        Non-unit IOP vectors should be normalized before use.
        Result should be identical to the normalized case.
        """
        # Slightly non-unit IOP (can happen with float rounding in DICOM)
        iop_nonunit = [1.0000001, 0.0, 0.0, 0.0, 0.9999999, 0.0]
        iop_unit    = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        slices_nonunit = [
            _make_slice([0, 0, 0], iop_nonunit, [2, 2], 5, 1),
            _make_slice([0, 0, 5], iop_nonunit, [2, 2], 5, 2),
        ]
        slices_unit = [
            _make_slice([0, 0, 0], iop_unit, [2, 2], 5, 1),
            _make_slice([0, 0, 5], iop_unit, [2, 2], 5, 2),
        ]

        affine_nonunit = compute_affine(slices_nonunit)
        affine_unit    = compute_affine(slices_unit)

        np.testing.assert_array_almost_equal(affine_nonunit, affine_unit, decimal=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)