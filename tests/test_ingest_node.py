
import unittest
from unittest.mock import MagicMock, patch
from fmri_preproc.io.nodes import DataIngestNode

class TestDataIngestNode(unittest.TestCase):
    
    @patch('fmri_preproc.io.nodes.BIDSDataset')
    def test_ingest_valid(self, mock_bids_cls):
        # Setup mock
        mock_ds = MagicMock()
        mock_bids_cls.return_value = mock_ds
        
        # Mock get_scans return
        mock_ds.get_scans.return_value = {
            'anat': [{'path': '/bids/sub-01/anat/sub-01_T1w.nii', 'entities': {}}],
            'func': [{'path': '/bids/sub-01/func/sub-01_task-rest_bold.nii', 'entities': {}}]
        }
        
        node = DataIngestNode("Ingest")
        node.set_input('bids_dir', '/fake/bids')
        node.set_input('subject', 'sub-01')
        
        node.run()
        
        # Verify outputs
        self.assertEqual(node.get_output('anat_file'), '/bids/sub-01/anat/sub-01_T1w.nii')
        self.assertEqual(len(node.get_output('func_files')), 1)
        self.assertEqual(node.get_output('subject_data')['subject'], 'sub-01')

    @patch('fmri_preproc.io.nodes.BIDSDataset')
    def test_ingest_missing(self, mock_bids_cls):
        mock_ds = MagicMock()
        mock_bids_cls.return_value = mock_ds
        mock_ds.get_scans.return_value = {'anat': [], 'func': []}
        
        node = DataIngestNode("IngestEmpty")
        node.set_input('bids_dir', '/fake/bids')
        node.set_input('subject', 'sub-02')
        
        node.run()
        
        self.assertIsNone(node.get_output('anat_file'))
        self.assertEqual(node.get_output('func_files'), [])

if __name__ == '__main__':
    unittest.main()
