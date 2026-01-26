import os
from bids import BIDSLayout

class BIDSDataset:
    """
    Wrapper around pybids BIDSLayout.
    """
    def __init__(self, bids_root: str):
        # Indexing can be slow for huge datasets; validate=False speeds it up
        try:
            self.layout = BIDSLayout(bids_root, validate=False)
        except Exception as e:
            print(f"Error initializing BIDS layout: {e}")
            self.layout = None

    def get_subjects(self):
        if not self.layout:
            return []
        return self.layout.get_subjects()

    def get_scans(self, subject: str):
        if not self.layout:
            return {'anat': [], 'func': []}
            
        scans = {'anat': [], 'func': []}
        
        # Anat (T1w)
        # return_type='file' gives paths, 'obj' gives BIDSFile objects
        t1s = self.layout.get(subject=subject, datatype='anat', suffix='T1w', extension=['nii', 'nii.gz'])
        for f in t1s:
            scans['anat'].append({
                'path': f.path, 
                'entities': f.entities,
                'meta': f.get_metadata()
            })

        # Func (BOLD)
        bolds = self.layout.get(subject=subject, datatype='func', suffix='bold', extension=['nii', 'nii.gz'])
        for f in bolds:
            scans['func'].append({
                'path': f.path, 
                'entities': f.entities,
                'meta': f.get_metadata()
            })
            
        return scans
