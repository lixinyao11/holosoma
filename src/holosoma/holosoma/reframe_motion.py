"""Reframe a holosoma-format motion NPZ clip to start at the origin facing +x."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma.utils.motion_reframe import (  # noqa: E402
    default_reframed_output_path,
    reframe_motion_dict,
)
from holosoma.utils.path import resolve_data_file_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reframe a holosoma-format motion NPZ clip to start at xy=(0,0) facing +x.")
    parser.add_argument("--input", required=True, help="Input motion NPZ path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path. Defaults to '<input>_reframed.npz' next to the resolved input file.",
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
    resolved_output = Path(args.output).expanduser().resolve() if args.output else default_reframed_output_path(args.input)

    if resolved_output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {resolved_output}. Use --overwrite to replace it.")

    with np.load(resolved_input, allow_pickle=False) as motion_data:
        reframed = reframe_motion_dict(motion_data)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(resolved_output, **reframed)

    print(f"Input:  {resolved_input}")
    print(f"Output: {resolved_output}")


if __name__ == "__main__":
    main()
