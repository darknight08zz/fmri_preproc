
import unittest
from unittest.mock import MagicMock, patch
import os
from fmri_preproc.func.nodes import DummyScanNode, RealignmentNode, SliceTimingNode, CoregistrationNode

class TestFuncNodes(unittest.TestCase):
    
    @patch('fmri_preproc.func.nodes.DummyScanRemoval')
    def test_dummy_scan_node(self, mock_cls):
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        mock_inst.run.return_value = True
        
        node = DummyScanNode("Dummy")
        node.set_input('bold_file', os.path.normpath('/data/sub-01_bold.nii.gz'))
        node.run()
        
        expected_out = os.path.normpath('/data/sub-01_bold_desc-dummy.nii.gz')
        self.assertEqual(node.get_output('trimmed_bold'), expected_out)
        mock_inst.run.assert_called_once()

    @patch('fmri_preproc.func.nodes.MotionCorrection')
    def test_realignment_node(self, mock_mc_cls):
        mock_mc = MagicMock()
        mock_mc_cls.return_value = mock_mc
        # Mock run return
        mock_mc.run.return_value = (os.path.normpath('/data/rsub-01_bold.nii.gz'), os.path.normpath('/data/rsub-01_bold.nii.gz.mat'))
        
        node = RealignmentNode("Realign")
        node.set_input('bold_file', os.path.normpath('/data/sub-01_bold.nii.gz'))
        
        # We need to mock os.path.exists/shutil for the mean file logic in execute
        with patch('os.path.exists', return_value=False), \
             patch('shutil.copy') as mock_copy:
            
            node.run()
            
            self.assertEqual(node.get_output('realigned_bold'), os.path.normpath('/data/rsub-01_bold.nii.gz'))
            self.assertEqual(node.get_output('motion_params'), os.path.normpath('/data/rsub-01_bold.nii.gz.par'))
            # Check mean file logic (placeholder generation)
            expected_mean = os.path.normpath('/data/meansub-01_bold.nii.gz')
            self.assertEqual(node.get_output('mean_bold'), expected_mean)

    @patch('fmri_preproc.func.nodes.SliceTiming')
    def test_stc_node(self, mock_stc_cls):
        mock_stc = MagicMock()
        mock_stc_cls.return_value = mock_stc
        mock_stc.run.return_value = True
        
        node = SliceTimingNode("STC")
        node.set_input('bold_file', os.path.normpath('/data/rsub-01_bold.nii.gz'))
        node.set_input('tr', 2.0)
        
        node.run()
        
        # Check output prefix 'a' applied to 'rsub...' -> 'arsub...'
        self.assertEqual(node.get_output('stc_bold'), os.path.normpath('/data/arsub-01_bold.nii.gz'))
        mock_stc.run.assert_called_once()

    @patch('fmri_preproc.func.nodes.Coregistration')
    def test_coreg_node(self, mock_coreg_cls):
        mock_coreg = MagicMock()
        mock_coreg_cls.return_value = mock_coreg
        # Mock run return (reg_img, mat)
        mock_coreg.run.return_value = (os.path.normpath('/data/coreg_sub-01_T1w.nii.gz'), os.path.normpath('/data/coreg_sub-01_T1w.mat'))
        
        node = CoregistrationNode("Coreg")
        node.set_input('ref_func', os.path.normpath('/data/mean_bold.nii.gz'))
        node.set_input('source_anat', os.path.normpath('/data/sub-01_T1w.nii.gz'))
        
        node.run()
        
        self.assertEqual(node.get_output('coreg_anat'), os.path.normpath('/data/coreg_sub-01_T1w.nii.gz'))
        self.assertEqual(node.get_output('coreg_matrix'), os.path.normpath('/data/coreg_sub-01_T1w.mat'))
        
        # Verify argument order: run(input=anat, reference=mean, ...)
        mock_coreg.run.assert_called_with(
            input_path=os.path.normpath('/data/sub-01_T1w.nii.gz'),
            reference_path=os.path.normpath('/data/mean_bold.nii.gz'),
            output_path=os.path.normpath('/data/coreg_sub-01_T1w.nii.gz'),
            wm_seg_path=None
        )

if __name__ == '__main__':
    unittest.main()
