from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fmri_preproc.io.bids import BIDSDataset
from fmri_preproc.validation.metadata import MetadataValidator, ValidationReport

router = APIRouter(prefix="/validation", tags=["validation"])

class ValidationRequest(BaseModel):
    bids_path: str
    subject: str

@router.post("/subject", response_model=ValidationReport)
async def validate_subject(req: ValidationRequest):
    try:
        ds = BIDSDataset(req.bids_path)
        scans = ds.get_scans(req.subject)
        
        validator = MetadataValidator()
        report = validator.validate(req.subject, scans)
        
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
