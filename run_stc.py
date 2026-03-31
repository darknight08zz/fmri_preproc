import argparse
import json
import nibabel as nib
from pathlib import Path
from stc.loader import VolumeLoader
from stc.correction import SliceTimer

def parse_slice_order(order_type, n_slices):
    if order_type == "ascending":
        return list(range(n_slices))

    elif order_type == "descending":
        return list(range(n_slices - 1, -1, -1))

    elif order_type == "interleaved":
        odds = list(range(0, n_slices, 2))
        evens = list(range(1, n_slices, 2))
        return odds + evens

    else:
        raise ValueError("Invalid slice order type")

def run_stc(input_file, tr=None, slice_order_type="ascending",
            ref_slice=0, ta=None):

    loader = VolumeLoader(input_file)
    data, affine, header, header_tr = loader.load()

    nifti_path = Path(input_file)
    json_path = nifti_path.with_suffix(".json")

    metadata = {}

    if json_path.exists():
        print(f"Found metadata JSON: {json_path}")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        if tr is None and "RepetitionTime" in metadata:
            tr = metadata["RepetitionTime"]
            print(f"Using TR from JSON: {tr} sec")

    if tr is None:
        if header_tr is None:
            raise ValueError("TR not provided and not found in header or JSON")
        tr = header_tr
        print(f"Using TR from NIfTI header: {tr} sec")

    n_slices = data.shape[2]

    if "SliceTiming" in metadata:
        print("Using SliceTiming vector from JSON")
        slice_times = metadata["SliceTiming"]
        if len(slice_times) != n_slices:
            raise ValueError("SliceTiming length does not match number of slices")

        timer = SliceTimer(
            data,
            tr,
            slice_order=None,
            ref_slice=ref_slice,
            ta=ta,
            slice_times=slice_times
        )

    else:
        print(f"No SliceTiming found. Using slice_order: {slice_order_type}")
        
        # Check if slice_order_type is a comma-separated list of numbers
        if "," in slice_order_type or slice_order_type.isdigit():
            try:
                slice_order = [int(x.strip()) for x in slice_order_type.split(",")]
                # SPM uses 1-based, our SliceTimer uses 0-based. 
                # The extractor (extract_stc_params.py) returns 1-based.
                # Let's adjust if they look 1-based.
                if max(slice_order) == n_slices and min(slice_order) == 1:
                    slice_order = [x - 1 for x in slice_order]
            except ValueError:
                raise ValueError(f"Could not parse numeric slice order: {slice_order_type}")
        else:
            slice_order = parse_slice_order(slice_order_type, n_slices)

        timer = SliceTimer(
            data,
            tr,
            slice_order=slice_order,
            ref_slice=ref_slice,
            ta=ta
        )

    corrected = timer.correct()
    output_path = loader.output_path()
    new_img = nib.Nifti1Image(corrected, affine, header)
    nib.save(new_img, output_path)

    result = {
        "status": "completed",
        "output": str(output_path)
    }
    print(json.dumps(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice Timing Correction with JSON auto-detection")
    parser.add_argument("input_file", help="Path to 4D NIfTI file")
    parser.add_argument("--tr", type=float, help="Repetition time (seconds)")
    parser.add_argument("--ta", type=float, help="Acquisition time (seconds)")
    parser.add_argument("--slice_order",
                        type=str,
                        default="ascending",
                        help="Slice acquisition order (choice or comma-separated numbers)")
    parser.add_argument("--ref_slice",
                        type=int,
                        default=0,
                        help="Reference slice index (0-based)")

    args = parser.parse_args()

    run_stc(
        args.input_file,
        tr=args.tr,
        slice_order_type=args.slice_order,
        ref_slice=args.ref_slice,
        ta=args.ta
    )
