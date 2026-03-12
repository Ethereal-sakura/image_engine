from __future__ import annotations

import argparse

from .engine import BasicColorRenderer
from .params import BasicColorParams



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RapidRAW Basic + Color only renderer (non-RAW)")
    parser.add_argument("--input", required=True, help="Input image path: tif/png/jpg")
    parser.add_argument("--params", required=True, help="JSON params file for Basic + Color only")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for JPG output")
    parser.add_argument("--save-intermediates-dir", default=None, help="Optional directory for 4 intermediate PNGs")
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    params = BasicColorParams.from_json_file(args.params)
    renderer = BasicColorRenderer()
    renderer.render_file(
        input_path=args.input,
        params=params,
        output_path=args.output,
        debug_dir=args.save_intermediates_dir,
        quality=int(args.quality),
    )
    print(f"Basic+Color rendered: {args.input} -> {args.output}")
    return 0
