
import unittest
from unittest.mock import MagicMock, patch
import os
from fmri_preproc.anat.nodes import SegmentationNode
from fmri_preproc.func.nodes import NormalizationNode, SmoothingNode

class TestAdvancedNodes(unittest.TestCase):
    
    @patch('fmri_preproc.anat.nodes.Segmentation')
    @patch('fmri_preproc.anat.nodes.WarpEstimation')
    def test_segmentation_node(self, mock_warp_cls, mock_fast_cls):
        # FAST setup
        mock_fast = MagicMock()
        mock_fast_cls.return_value = mock_fast
        mock_fast.run.return_value = True
        
        # Warp setup
        mock_warp = MagicMock()
        mock_warp_cls.return_value = mock_warp
        # Warp returns (warped_img, [transforms])
        mock_warp.run.return_value = (os.path.normpath('/data/y_sub-01_T1w_Warped.nii.gz'), [os.path.normpath('/data/y_sub-01_T1w_1Warp.nii.gz'), os.path.normpath('/data/y_sub-01_T1w_0GenericAffine.mat')])
        
        node = SegmentationNode("Seg")
        node.set_input('anat_file', os.path.normpath('/data/coreg_sub-01_T1w.nii.gz'))
        
        node.run()
        
        # Verify outputs
        self.assertIn(os.path.normpath('/data/y_sub-01_T1w_1Warp.nii.gz'), node.get_output('forward_transforms'))
        expected_def = os.path.normpath('/data/y_sub-01_T1w_1Warp.nii.gz')
        self.assertEqual(node.get_output('deformation_field'), expected_def)
        # Check bias corrected output
        self.assertEqual(node.get_output('bias_corrected'), os.path.normpath('/data/coreg_sub-01_T1w_restore.nii.gz'))

    @patch('fmri_preproc.func.nodes.SpatialTransforms')
    def test_normalization_node(self, mock_xform_cls):
        mock_xform = MagicMock()
        mock_xform_cls.return_value = mock_xform
        mock_xform.run.return_value = True
        
        node = NormalizationNode("Norm")
        node.set_input('bold_file', os.path.normpath('/data/arsub-01_bold.nii.gz'))
        node.set_input('deformation_field', [os.path.normpath('/data/y_warp.nii.gz')])
        node.set_input('additional_transforms', [os.path.normpath('/data/coreg.mat')])
        
        node.run()
        
        self.assertEqual(node.get_output('normalized_bold'), os.path.normpath('/data/warsub-01_bold.nii.gz'))
        
        # Verify call arguments
        # args: (input, ref, out, transforms)
        # transforms should be [warp, coreg]
        mock_xform.run.assert_called_with(
            os.path.normpath('/data/arsub-01_bold.nii.gz'),
            "templates/MNI152_T1_2mm.nii.gz",
            os.path.normpath('/data/warsub-01_bold.nii.gz'),
            [os.path.normpath('/data/y_warp.nii.gz'), os.path.normpath('/data/coreg.mat')]
        )

    @patch('fmri_preproc.func.nodes.SpatialSmoothing')
    def test_smoothing_node(self, mock_smooth_cls):
        mock_smooth = MagicMock()
        mock_smooth_cls.return_value = mock_smooth
        mock_smooth.run.return_value = "dummy"
        
        node = SmoothingNode("Smooth")
        node.set_input('bold_file', os.path.normpath('/data/warsub-01_bold.nii.gz'))
        node.set_input('fwhm', [6,6,6])
        
        node.run()
        
        self.assertEqual(node.get_output('smoothed_bold'), os.path.normpath('/data/swarsub-01_bold.nii.gz'))
        
        mock_smooth.run.assert_called_with(
             os.path.normpath('/data/warsub-01_bold.nii.gz'),
             os.path.normpath('/data/swarsub-01_bold.nii.gz'),
             fwhm=[6,6,6]
        )

if __name__ == '__main__':
    unittest.main()
