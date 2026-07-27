"""Mirror a holosoma-format motion NPZ clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma.utils.motion_mirror import (  # noqa: E402
    default_mirrored_output_path,
    infer_urdf_path_from_motion_path,
    mirror_motion_dict,
)
from holosoma.utils.path import resolve_data_file_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a holosoma-format motion NPZ clip.")
    parser.add_argument("--input", required=True, help="Input motion NPZ path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path. Defaults to '<input>_mirrored.npz' next to the resolved input file.",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="Optional URDF used to infer joint sign conventions. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--mirror-axis",
        default="y",
        choices=("y",),
        help="World-space axis normal of the mirror plane. Only 'y' left/right mirroring is currently supported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_input = Path(resolve_data_file_path(args.input))
    resolved_output = Path(args.output).expanduser().resolve() if args.output else default_mirrored_output_path(args.input)
    urdf_path = args.urdf or infer_urdf_path_from_motion_path(args.input)

    if resolved_output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {resolved_output}. Use --overwrite to replace it.")

    with np.load(resolved_input, allow_pickle=False) as motion_data:
        mirrored = mirror_motion_dict(
            motion_data,
            mirror_axis=args.mirror_axis,
            urdf_path=urdf_path,
            motion_path=args.input,
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(resolved_output, **mirrored)

    print(f"Input:  {resolved_input}")
    print(f"Output: {resolved_output}")
    if urdf_path is not None:
        print(f"URDF:   {resolve_data_file_path(urdf_path)}")
    else:
        print("URDF:   not provided; used built-in joint-name heuristic")


if __name__ == "__main__":
    main()
