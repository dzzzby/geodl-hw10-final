#!/usr/bin/env python3
"""Create a two-color overlay PLY for visual registration checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluate_reconstruction import read_ply, random_sample, write_ply


def parse_color(text: str) -> np.ndarray:
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 3 or any(x < 0 or x > 255 for x in parts):
        raise argparse.ArgumentTypeError("Color must be R,G,B with values in [0,255]")
    return np.array(parts, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True, help="Aligned prediction PLY")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth PLY")
    parser.add_argument("--out", type=Path, required=True, help="Output overlay PLY")
    parser.add_argument("--sample-pred", type=int, default=0, help="0 means keep all pred points")
    parser.add_argument("--sample-gt", type=int, default=0, help="0 means keep all GT points")
    parser.add_argument("--pred-color", type=parse_color, default=parse_color("255,80,60"))
    parser.add_argument("--gt-color", type=parse_color, default=parse_color("60,140,255"))
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = read_ply(args.pred).xyz
    gt = read_ply(args.gt).xyz

    if args.sample_pred and args.sample_pred > 0:
        pred = random_sample(pred, args.sample_pred, args.seed)
    if args.sample_gt and args.sample_gt > 0:
        gt = random_sample(gt, args.sample_gt, args.seed + 1)

    points = np.concatenate([pred, gt], axis=0)
    colors = np.concatenate([
        np.tile(args.pred_color[None, :], (len(pred), 1)),
        np.tile(args.gt_color[None, :], (len(gt), 1)),
    ], axis=0)
    write_ply(args.out, points, colors)
    print(f"wrote {args.out} with {len(pred):,} pred points and {len(gt):,} gt points")


if __name__ == "__main__":
    main()
