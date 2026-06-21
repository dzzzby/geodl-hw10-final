#!/usr/bin/env python3
"""Evaluate a reconstructed point cloud against a GT PLY.

This script is intentionally headless: it can run on a remote server without
CloudCompare/Open3D. It supports LingBot-Map GLB exports by extracting the
largest point cloud geometry from the GLB.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray | None = None


def read_ply(path: Path) -> PointCloud:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        fmt = next(line for line in header_lines if line.startswith("format ")).split()[1]
        vertex_line = next(line for line in header_lines if line.startswith("element vertex "))
        vertex_count = int(vertex_line.split()[2])

        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("element "):
                in_vertex = line.startswith("element vertex ")
                continue
            if in_vertex and line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("List properties in vertex elements are not supported")
                properties.append((parts[2], PLY_TYPES[parts[1]]))

        if fmt == "binary_little_endian":
            arr = np.fromfile(f, dtype=np.dtype(properties), count=vertex_count)
            xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
            rgb = None
            if {"red", "green", "blue"}.issubset(arr.dtype.names or ()):
                rgb = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.uint8)
            return PointCloud(xyz=xyz, rgb=rgb)

        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                rows.append(f.readline().decode("ascii").split())
            arr = np.array(rows, dtype=np.float64)
            names = [name for name, _ in properties]
            xyz = arr[:, [names.index("x"), names.index("y"), names.index("z")]]
            rgb = None
            if {"red", "green", "blue"}.issubset(names):
                rgb = arr[:, [names.index("red"), names.index("green"), names.index("blue")]].astype(np.uint8)
            return PointCloud(xyz=xyz, rgb=rgb)

    raise ValueError(f"Unsupported PLY format: {fmt}")


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    if rgb is None:
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
        arr = np.empty(len(xyz), dtype=dtype)
    else:
        rgb = np.asarray(rgb, dtype=np.uint8)
        dtype = np.dtype([
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        arr = np.empty(len(xyz), dtype=dtype)
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    with path.open("wb") as f:
        props = [
            "property float x",
            "property float y",
            "property float z",
        ]
        if rgb is not None:
            props += ["property uchar red", "property uchar green", "property uchar blue"]
        header = "\n".join([
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(xyz)}",
            *props,
            "end_header\n",
        ])
        f.write(header.encode("ascii"))
        arr.tofile(f)


def load_glb_pointcloud(path: Path) -> PointCloud:
    scene = trimesh.load(path, force="scene")
    clouds = []
    for geom in scene.geometry.values():
        if type(geom).__name__ == "PointCloud":
            clouds.append(geom)
    if not clouds:
        raise ValueError(f"No PointCloud geometry found in {path}")
    cloud = max(clouds, key=lambda g: len(g.vertices))
    xyz = np.asarray(cloud.vertices, dtype=np.float64)
    colors = getattr(cloud.visual, "vertex_colors", None)
    rgb = None
    if colors is not None and len(colors) == len(xyz):
        rgb = np.asarray(colors[:, :3], dtype=np.uint8)
    return PointCloud(xyz=xyz, rgb=rgb)


def load_cloud(path: Path) -> PointCloud:
    if path.suffix.lower() == ".glb":
        return load_glb_pointcloud(path)
    if path.suffix.lower() == ".ply":
        return read_ply(path)
    raise ValueError(f"Unsupported point cloud input: {path}")


def load_mesh_surface(path: Path, samples: int, seed: int) -> PointCloud:
    mesh_or_scene = trimesh.load(path, process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [
            geom for geom in mesh_or_scene.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No triangle mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = mesh_or_scene
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Input is not a triangle mesh: {path}")
    np.random.seed(seed)
    xyz, _ = trimesh.sample.sample_surface(mesh, samples)
    return PointCloud(xyz=np.asarray(xyz, dtype=np.float64))


def finite_cloud(cloud: PointCloud) -> PointCloud:
    mask = np.isfinite(cloud.xyz).all(axis=1)
    rgb = cloud.rgb[mask] if cloud.rgb is not None else None
    return PointCloud(xyz=cloud.xyz[mask], rgb=rgb)


def random_sample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(points) <= n:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def robust_extent(points: np.ndarray) -> np.ndarray:
    return np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)


def pca_frame(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    frame = vecs[:, ::-1]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return frame


def similarity_transform(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_fix[-1, -1] = -1
    rot = u @ s_fix @ vt
    scale = 1.0
    if with_scale:
        var_src = np.mean(np.sum(src_c * src_c, axis=1))
        scale = float(np.trace(np.diag(singular) @ s_fix) / max(var_src, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    mat = np.eye(4)
    mat[:3, :3] = scale * rot
    mat[:3, 3] = trans
    return mat


def transform_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return points @ mat[:3, :3].T + mat[:3, 3]


def proper_axis_mappings(include_permutations: bool) -> list[np.ndarray]:
    if not include_permutations:
        return [
            np.diag([sx, sy, sz])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
            if sx * sy * sz > 0
        ]

    mats = []
    for perm in itertools.permutations(range(3)):
        perm_mat = np.eye(3)[:, perm]
        for signs in itertools.product((-1, 1), repeat=3):
            mat = perm_mat @ np.diag(signs)
            if np.linalg.det(mat) > 0:
                mats.append(mat)
    return mats


def initial_pca_candidates(src: np.ndarray, dst: np.ndarray, include_permutations: bool) -> list[np.ndarray]:
    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)
    src_frame = pca_frame(src)
    dst_frame = pca_frame(dst)
    src_extent = np.linalg.norm(robust_extent(src))
    dst_extent = np.linalg.norm(robust_extent(dst))
    scale = dst_extent / max(src_extent, 1e-12)

    mats = []
    for axis_map in proper_axis_mappings(include_permutations):
        rot = dst_frame @ axis_map @ src_frame.T
        mat = np.eye(4)
        mat[:3, :3] = scale * rot
        mat[:3, 3] = dst_center - scale * (rot @ src_center)
        mats.append(mat)
    return mats


def trimmed_similarity_icp(
    src: np.ndarray,
    dst: np.ndarray,
    init: np.ndarray,
    iterations: int,
    trim_fraction: float,
    with_scale: bool,
    workers: int,
) -> tuple[np.ndarray, float]:
    tree = cKDTree(dst)
    mat = init.copy()
    keep = max(32, int(len(src) * trim_fraction))
    score = float("inf")
    for _ in range(iterations):
        moved = transform_points(src, mat)
        dist, idx = tree.query(moved, workers=workers)
        order = np.argpartition(dist, keep - 1)[:keep]
        delta = similarity_transform(moved[order], dst[idx[order]], with_scale=with_scale)
        mat = delta @ mat
        score = float(np.median(dist[order]))
    return mat, score


def trimmed_nn_median(source: np.ndarray, target: np.ndarray, trim_fraction: float, workers: int) -> float:
    tree = cKDTree(target)
    dist, _ = tree.query(source, workers=workers)
    keep = max(32, int(len(source) * trim_fraction))
    keep = min(keep, len(source))
    order = np.argpartition(dist, keep - 1)[:keep]
    return float(np.median(dist[order]))


def bidirectional_trimmed_score(
    pred: np.ndarray,
    gt: np.ndarray,
    mat: np.ndarray,
    trim_fraction: float,
    workers: int,
) -> tuple[float, float, float]:
    moved = transform_points(pred, mat)
    pred_to_gt = trimmed_nn_median(moved, gt, trim_fraction, workers=workers)
    gt_to_pred = trimmed_nn_median(gt, moved, trim_fraction, workers=workers)
    return (pred_to_gt + gt_to_pred) / 2.0, pred_to_gt, gt_to_pred


def auto_align(
    pred: np.ndarray,
    gt: np.ndarray,
    sample_points: int,
    iterations: int,
    trim_fraction: float,
    seed: int,
    with_scale: bool,
    pca_permutations: bool,
    workers: int,
    extra_inits: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    pred_s = random_sample(pred, sample_points, seed)
    gt_s = random_sample(gt, sample_points, seed + 1)
    candidates = initial_pca_candidates(pred_s, gt_s, include_permutations=pca_permutations)
    if extra_inits:
        candidates = [init.copy() for init in extra_inits] + candidates
    best_mat = candidates[0]
    best_score = float("inf")
    best_forward = float("inf")
    best_backward = float("inf")
    for i, init in enumerate(candidates):
        print(f"align candidate {i + 1}/{len(candidates)}...", flush=True)
        mat, _ = trimmed_similarity_icp(
            pred_s, gt_s, init, iterations=iterations,
            trim_fraction=trim_fraction, with_scale=with_scale, workers=workers,
        )
        score, forward, backward = bidirectional_trimmed_score(
            pred_s, gt_s, mat, trim_fraction, workers=workers,
        )
        print(
            f"  score={score:.6g} pred->gt={forward:.6g} gt->pred={backward:.6g}",
            flush=True,
        )
        if score < best_score:
            best_mat = mat
            best_score = score
            best_forward = forward
            best_backward = backward
    return best_mat, {
        "icp_sample_bidirectional_median": best_score,
        "icp_sample_pred_to_gt_median": best_forward,
        "icp_sample_gt_to_pred_median": best_backward,
        "num_candidates": len(candidates),
        "pca_permutations": pca_permutations,
        "num_extra_init_transforms": 0 if extra_inits is None else len(extra_inits),
        "workers": workers,
    }


def distances(source: np.ndarray, target: np.ndarray, chunk: int = 1_000_000, workers: int = 1) -> np.ndarray:
    tree = cKDTree(target)
    out = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk):
        end = min(start + chunk, len(source))
        print(f"  distance chunk {start:,}:{end:,}", flush=True)
        out[start:end], _ = tree.query(source[start:end], workers=workers)
    return out


def stats(dist: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "rmse": float(np.sqrt(np.mean(dist * dist))),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
        "within_10cm_percent": float(np.mean(dist <= 0.10) * 100.0),
        "within_20cm_percent": float(np.mean(dist <= 0.20) * 100.0),
    }


def distance_colors(dist: np.ndarray, max_dist: float) -> np.ndarray:
    t = np.clip(dist / max_dist, 0.0, 1.0)
    rgb = np.zeros((len(dist), 3), dtype=np.uint8)
    rgb[:, 0] = (255 * t).astype(np.uint8)
    rgb[:, 1] = (255 * (1.0 - np.abs(t - 0.5) * 2.0)).astype(np.uint8)
    rgb[:, 2] = (255 * (1.0 - t)).astype(np.uint8)
    return rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True, help="Predicted .glb or .ply")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth .ply")
    parser.add_argument("--out-dir", type=Path, default=Path("eval_outputs/Sequence_01"))
    parser.add_argument("--pred-as-mesh", action="store_true", help="Sample --pred as a triangle mesh surface")
    parser.add_argument("--gt-as-mesh", action="store_true", help="Sample --gt as a triangle mesh surface")
    parser.add_argument("--mesh-sample", type=int, default=2_000_000, help="Number of surface samples for mesh inputs")
    parser.add_argument("--no-align", action="store_true", help="Only compute metrics in the input coordinate frames")
    parser.add_argument("--transform", type=Path, help="Optional 4x4 matrix mapping prediction to GT coordinates")
    parser.add_argument(
        "--init-transform",
        type=Path,
        action="append",
        default=[],
        help="Optional 4x4 matrix to use as an additional ICP initialization; can be repeated.",
    )
    parser.add_argument("--no-scale", action="store_true", help="Disable scale updates during auto alignment")
    parser.add_argument("--sample-align", type=int, default=100_000)
    parser.add_argument(
        "--pca-permutations",
        action="store_true",
        help="Try all 24 right-handed PCA axis permutations instead of only 4 sign flips.",
    )
    parser.add_argument(
        "--sample-eval",
        type=int,
        default=1_000_000,
        help=(
            "Sample this many query/source points for each direction. "
            "Targets stay full by default; use 0 for full query clouds too."
        ),
    )
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--trim-fraction", type=float, default=0.70)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Workers for scipy cKDTree queries. Default 1 avoids native crashes "
            "seen with highly parallel large-cloud queries; use -1 to request all cores."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_cloud = finite_cloud(
        load_mesh_surface(args.pred, args.mesh_sample, args.seed)
        if args.pred_as_mesh else load_cloud(args.pred)
    )
    gt_cloud = finite_cloud(
        load_mesh_surface(args.gt, args.mesh_sample, args.seed + 1)
        if args.gt_as_mesh else load_cloud(args.gt)
    )
    print(f"pred points: {len(pred_cloud.xyz):,}")
    print(f"gt points:   {len(gt_cloud.xyz):,}")

    write_ply(args.out_dir / "pred_extracted.ply", pred_cloud.xyz, pred_cloud.rgb)

    if args.transform:
        transform = np.loadtxt(args.transform)
        align_info = {"mode": "provided_transform"}
    elif args.no_align:
        transform = np.eye(4)
        align_info = {"mode": "none"}
    else:
        extra_inits = [np.loadtxt(path) for path in args.init_transform]
        transform, align_info = auto_align(
            pred_cloud.xyz, gt_cloud.xyz,
            sample_points=args.sample_align,
            iterations=args.icp_iterations,
            trim_fraction=args.trim_fraction,
            seed=args.seed,
            with_scale=not args.no_scale,
            pca_permutations=args.pca_permutations,
            workers=args.workers,
            extra_inits=extra_inits,
        )
        align_info["mode"] = "pca_trimmed_similarity_icp"

    pred_aligned = transform_points(pred_cloud.xyz, transform)
    write_ply(args.out_dir / "pred_aligned.ply", pred_aligned, pred_cloud.rgb)
    np.savetxt(args.out_dir / "pred_to_gt_transform.txt", transform, fmt="%.10g")

    pred_eval = pred_aligned
    gt_eval = gt_cloud.xyz
    if args.sample_eval and args.sample_eval > 0:
        pred_eval = random_sample(pred_eval, args.sample_eval, args.seed + 2)
        gt_eval = random_sample(gt_eval, args.sample_eval, args.seed + 3)

    print(f"computing pred -> gt distances ({len(pred_eval):,} queries -> {len(gt_cloud.xyz):,} target points)...")
    pred_to_gt = distances(pred_eval, gt_cloud.xyz, workers=args.workers)
    print(f"computing gt -> pred distances ({len(gt_eval):,} queries -> {len(pred_aligned):,} target points)...")
    gt_to_pred = distances(gt_eval, pred_aligned, workers=args.workers)

    result = {
        "pred": str(args.pred),
        "gt": str(args.gt),
        "num_pred_points": int(len(pred_cloud.xyz)),
        "num_gt_points": int(len(gt_cloud.xyz)),
        "num_pred_eval_points": int(len(pred_eval)),
        "num_gt_eval_points": int(len(gt_eval)),
        "num_pred_target_points": int(len(pred_aligned)),
        "num_gt_target_points": int(len(gt_cloud.xyz)),
        "alignment": align_info,
        "pred_to_gt": stats(pred_to_gt),
        "gt_to_pred": stats(gt_to_pred),
        "chamfer_mean": float((np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2.0),
        "transform_pred_to_gt": transform.tolist(),
    }

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    write_ply(args.out_dir / "pred_eval_colored_by_error.ply", pred_eval, distance_colors(pred_to_gt, 0.20))
    write_ply(args.out_dir / "gt_eval_colored_by_error.ply", gt_eval, distance_colors(gt_to_pred, 0.20))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
