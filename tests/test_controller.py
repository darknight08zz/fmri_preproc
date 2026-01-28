
import unittest
import os
from fmri_preproc.core.controller import PipelineController
from fmri_preproc.core.workflow import Workflow

class TestPipelineController(unittest.TestCase):
    def test_build_workflow(self):
        # We don't run the heavy pipeline, just build it and check graph structure
        controller = PipelineController(output_root="/tmp/out")
        wf = controller.build_workflow(subject="sub-test", bids_dir="/tmp/bids")
        
        self.assertIsInstance(wf, Workflow)
        self.assertEqual(wf.name, "Workflow_sub-test")
        
        # Check Node presence
        node_names = wf.nodes.keys()
        expected = [
            "Ingest", "Dummy", "Realign", "STC", 
            "Coreg", "Segmentation", "Normalization", 
            "Smoothing", "QC"
        ]
        for name in expected:
            self.assertIn(name, node_names)
            
        # Check some connections
        # e.g. Ingest -> Dummy
        connected = False
        for edge in wf.edges:
            # (src, src_out, dst, dst_in)
            if edge[0] == "Ingest" and edge[2] == "Dummy":
                connected = True
                break
        self.assertTrue(connected, "Ingest should be connected to Dummy")
        
        # Realignment -> STC
        connected = False
        for edge in wf.edges:
             if edge[0] == "Realign" and edge[2] == "STC":
                 connected = True
                 break
        self.assertTrue(connected, "Realign should be connected to STC")

if __name__ == '__main__':
    unittest.main()
