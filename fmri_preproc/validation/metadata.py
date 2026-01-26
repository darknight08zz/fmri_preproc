from typing import Dict, List, Any, Optional
import os
from pydantic import BaseModel


class ValidationIssue(BaseModel):
    severity: str # "critical", "warning"
    message: str
    scan_path: str

class ValidationReport(BaseModel):
    subject: str
    valid: bool
    issues: List[ValidationIssue]
    pipeline_config: Dict[str, bool]

from fmri_preproc.validation.orientation import check_orientation

class MetadataValidator:
    """
    Validates BIDS metadata to determine pipeline steps.
    """
    def validate(self, subject: str, scans: Dict[str, List[Dict]]) -> ValidationReport:
        issues = []
        config = {
            "slice_timing": False,
            "susceptibility_distortion": False,
            "motion_correction": True, # Always run unless critical failure
            "force_reorient": False
        }
        
        # Check T1w
        t1_scans = scans.get("anat", [])
        if not t1_scans:
            issues.append(ValidationIssue(
                severity="warning",
                message="No T1w anatomical image found. Processing in Functional-Only logic (Native Space).",
                scan_path="N/A"
            ))
        else:
            # Check orientation for T1
            for t1 in t1_scans:
                path = t1.get("path")
                if path and os.path.exists(path):
                    ornt, warns = check_orientation(path)
                    if ornt != "RAS":
                        config["force_reorient"] = True
                        issues.append(ValidationIssue(
                            severity="warning",
                            message=f"T1w orientation is {ornt} (not RAS). Pipeline will reorient.",
                            scan_path=path
                        ))
                    for w in warns:
                         issues.append(ValidationIssue(severity="warning", message=w, scan_path=path))

        
        # Check BOLD
        func_scans = scans.get("func", [])
        if not func_scans:
            issues.append(ValidationIssue(
                severity="critical",
                message="No BOLD functional images found.",
                scan_path="N/A"
            ))
            
        for scan in func_scans:
            meta = scan.get("meta", {})
            path = scan.get("path", "unknown")
            
            # RepetitionTime
            if "RepetitionTime" not in meta:
                 issues.append(ValidationIssue(
                    severity="critical",
                    message="Missing RepetitionTime (TR) in JSON sidecar.",
                    scan_path=path
                ))
            
            # SliceTiming
            if "SliceTiming" in meta:
                # If at least one scan has splice timing, we can enable it (conceptually)
                # In reality, needs to be per-run. But for this high-level check:
                config["slice_timing"] = True
            else:
                 issues.append(ValidationIssue(
                    severity="warning",
                    message="Missing SliceTiming. Skip slice timing correction.",
                    scan_path=path
                ))
                
            # PhaseEncodingDirection
            if "PhaseEncodingDirection" not in meta:
                issues.append(ValidationIssue(
                    severity="warning",
                    message="Missing PhaseEncodingDirection. SDC might be limited.",
                    scan_path=path
                ))
                
        valid = not any(i.severity == "critical" for i in issues)
        
        return ValidationReport(
            subject=subject,
            valid=valid,
            issues=issues,
            pipeline_config=config
        )
