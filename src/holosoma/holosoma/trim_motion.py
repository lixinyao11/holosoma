"""Trim a holosoma-format motion NPZ clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma.utils.motion_trim import (  # noqa: E402
    default_trimmed_output_path,
    resolve_trim_bounds,
    trim_motion_dict,
)
from holosoma.utils.path import resolve_data_file_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim a holosoma-format motion NPZ clip.")
    parser.add_argument("--input", required=True, help="Input motion NPZ path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path. Defaults to '<input>_trim.npz' next to the resolved input file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inclusive start frame. Negative values count back from the end.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end frame. Negative values count back from the end; for example, -10 drops the last 10 frames.",
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
    resolved_output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_trimmed_output_path(args.input)
    )

    if resolved_output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {resolved_output}. Use --overwrite to replace it.")

    with np.load(resolved_input, allow_pickle=False) as motion_data:
        num_frames = int(motion_data["joint_pos"].shape[0])
        start_idx, end_idx = resolve_trim_bounds(num_frames, start=args.start, end=args.end)
        trimmed = trim_motion_dict(motion_data, start=args.start, end=args.end)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(resolved_output, **trimmed)

    print(f"Input:  {resolved_input}")
    print(f"Output: {resolved_output}")
    print(f"Frames: {num_frames} -> {end_idx - start_idx} [{start_idx}, {end_idx})")


if __name__ == "__main__":
    main()
