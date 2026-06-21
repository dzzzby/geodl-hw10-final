#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# 
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from os import makedirs
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.dataset_readers import _find_colmap_sparse_dir
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal
from utils.read_write_model import read_images_binary, read_images_text

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def write_ply(path, xyz, rgb=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)

    if rgb is None:
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
        vertices = np.empty(len(xyz), dtype=dtype)
    else:
        rgb = np.asarray(rgb, dtype=np.uint8)
        dtype = np.dtype([
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        vertices = np.empty(len(xyz), dtype=dtype)
        vertices["red"] = rgb[:, 0]
        vertices["green"] = rgb[:, 1]
        vertices["blue"] = rgb[:, 2]

    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]

    props = [
        "property float x",
        "property float y",
        "property float z",
    ]
    if rgb is not None:
        props.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(xyz)}",
        *props,
        "end_header\n",
    ])

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def voxel_downsample(xyz, rgb=None, voxel_size=0.0):
    if voxel_size <= 0 or len(xyz) == 0:
        return xyz, rgb

    coords = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)

    xyz_down = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(xyz_down, inverse, xyz)
    xyz_down /= counts[:, None]

    rgb_down = None
    if rgb is not None:
        rgb_sum = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(rgb_sum, inverse, rgb.astype(np.float64))
        rgb_down = np.clip(np.round(rgb_sum / counts[:, None]), 0, 255).astype(np.uint8)

    return xyz_down.astype(np.float32), rgb_down


def random_subsample(xyz, rgb=None, max_points=0, seed=0):
    if max_points <= 0 or len(xyz) <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(xyz), size=max_points, replace=False)
    xyz = xyz[keep]
    rgb = rgb[keep] if rgb is not None else None
    return xyz, rgb


def view_name_stem(view):
    image_name = view.image_name
    if isinstance(image_name, (list, tuple)):
        image_name = "".join(str(part) for part in image_name)
    return Path(str(image_name)).stem


def view_device(view):
    for attr in ("original_image", "gt_image", "world_view_transform", "camera_center"):
        value = getattr(view, attr, None)
        if torch.is_tensor(value):
            return value.device
    data_device = getattr(view, "data_device", None)
    if data_device is not None:
        return torch.device(data_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_precomputed_depth(view, rendered_depth_dir):
    if not rendered_depth_dir:
        return None

    depth_dir = Path(rendered_depth_dir)
    stem = view_name_stem(view)
    candidates = [
        depth_dir / f"{stem}.npy",
        depth_dir / f"{stem}.npz",
        depth_dir / f"{stem}.png",
        depth_dir / f"{stem}.exr",
    ]
    for depth_path in candidates:
        if not depth_path.exists():
            continue
        if depth_path.suffix == ".npy":
            depth = np.load(depth_path)
        elif depth_path.suffix == ".npz":
            data = np.load(depth_path)
            key = "depth" if "depth" in data else data.files[0]
            depth = data[key]
        else:
            import cv2
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)

        depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth in {depth_path}, got shape {depth.shape}")
        return torch.from_numpy(depth).to(view_device(view))

    return None


def load_precomputed_color(view, rendered_color_dir):
    if not rendered_color_dir:
        return None

    import cv2

    color_dir = Path(rendered_color_dir)
    stem = view_name_stem(view)
    candidates = [
        color_dir / f"{stem}.jpg",
        color_dir / f"{stem}.png",
        color_dir / f"{stem}.jpeg",
        color_dir / f"{stem}.npy",
    ]
    for color_path in candidates:
        if not color_path.exists():
            continue
        if color_path.suffix == ".npy":
            color = np.load(color_path)
        else:
            color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if color is None:
                continue
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = np.asarray(color)
        if color.ndim != 3:
            raise ValueError(f"Expected HxWxC color in {color_path}, got shape {color.shape}")
        if color.shape[0] == 3 and color.shape[-1] != 3:
            color = np.moveaxis(color, 0, -1)
        if color.shape[-1] > 3:
            color = color[..., :3]
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0
        else:
            color = color.astype(np.float32)
            if color.max(initial=0.0) > 1.0:
                color /= 255.0
        return torch.from_numpy(color).permute(2, 0, 1).to(view_device(view))

    return None


def infer_rendered_depth_dir(output_root, set_name, iteration, args):
    if args.rendered_depth_root:
        depth_root = Path(args.rendered_depth_root).expanduser()
        if not depth_root.is_dir():
            raise FileNotFoundError(f"--rendered_depth_root is not a directory: {depth_root}")
        return depth_root

    out_dir = Path(output_root) / set_name / f"ours_{iteration}"
    candidates = [
        out_dir / "renders_depth",
        out_dir / "depth",
        out_dir / "depths",
        out_dir / "rendered_depth",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def infer_rendered_color_dir(output_root, set_name, iteration, args):
    if args.rendered_color_root:
        color_root = Path(args.rendered_color_root).expanduser()
        if not color_root.is_dir():
            raise FileNotFoundError(f"--rendered_color_root is not a directory: {color_root}")
        return color_root

    out_dir = Path(output_root) / set_name / f"ours_{iteration}"
    candidates = [
        out_dir / "renders",
        out_dir / "render",
        out_dir / "colors",
        out_dir / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def resize_depth_like_image(depth, image):
    target_hw = image.shape[-2:]
    if tuple(depth.shape[-2:]) == tuple(target_hw):
        return depth
    depth = depth[None, None]
    depth = torch.nn.functional.interpolate(depth, size=target_hw, mode="nearest")
    return depth[0, 0]


def resize_color_like_depth(image, depth):
    target_hw = depth.shape[-2:]
    if tuple(image.shape[-2:]) == tuple(target_hw):
        return image
    image = image[None]
    image = torch.nn.functional.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)
    return image[0]


def backproject_depth(view, depth, image, stride=1, min_depth=0.0, max_depth=0.0):
    if stride > 1:
        depth = depth[::stride, ::stride]
        image = image[:, ::stride, ::stride]

    height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32),
        torch.arange(width, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )

    xs = xs * stride
    ys = ys * stride
    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    cx = view.image_width * 0.5
    cy = view.image_height * 0.5

    valid = torch.isfinite(depth) & (depth > 0)
    if min_depth > 0:
        valid &= depth >= min_depth
    if max_depth > 0:
        valid &= depth <= max_depth

    z = depth[valid]
    if z.numel() == 0:
        return None, None

    x = (xs[valid] - cx) / fx * z
    y = (ys[valid] - cy) / fy * z
    ones = torch.ones_like(z)
    cam_points = torch.stack((x, y, z, ones), dim=1)

    c2w = torch.inverse(view.world_view_transform)
    world_points = cam_points @ c2w
    colors = image.permute(1, 2, 0)[valid]

    return world_points[:, :3].detach().cpu().numpy(), colors.detach().cpu().numpy()


def render_view_depth_color(
    view,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    opacity_threshold,
    depth_source="auto",
    rendered_depth_dir=None,
    rendered_color_dir=None,
    return_confidence=False,
):
    precomputed_depth = None
    if depth_source in {"auto", "precomputed"}:
        precomputed_depth = load_precomputed_depth(view, rendered_depth_dir)
        if depth_source == "precomputed" and precomputed_depth is None:
            raise FileNotFoundError(
                f"No precomputed depth found for {view_name_stem(view)} in {rendered_depth_dir}"
            )

    if precomputed_depth is not None:
        precomputed_color = load_precomputed_color(view, rendered_color_dir)
        if precomputed_color is not None:
            depth = precomputed_depth
            image = resize_color_like_depth(precomputed_color, depth).clamp(0.0, 1.0)
            confidence = (torch.isfinite(depth) & (depth > 0)).to(image.dtype)
            if return_confidence:
                return depth, image, confidence
            return depth, image

    pkg = render(
        view,
        gaussians,
        pipeline,
        background,
        use_trained_exp=train_test_exp,
        separate_sh=separate_sh,
    )

    image = pkg["render"].clamp(0.0, 1.0)

    if precomputed_depth is not None:
        depth = resize_depth_like_image(precomputed_depth, image)
        confidence = (torch.isfinite(depth) & (depth > 0)).to(image.dtype)
    elif depth_source in {"auto", "planargs", "planargs_plane", "plane"} and "plane_depth" in pkg:
        depth = pkg["plane_depth"].squeeze()
        confidence = (
            pkg.get("rendered_distance", torch.ones_like(depth))
            .squeeze()
            .detach()
            .isfinite()
            .to(image.dtype)
        )
    elif depth_source in {"auto", "3dgs", "inverse"} and "depth" in pkg:
        white = torch.ones_like(gaussians.get_xyz, device="cuda")
        opacity_pkg = render(
            view,
            gaussians,
            pipeline,
            torch.zeros(3, dtype=torch.float32, device="cuda"),
            override_color=white,
            separate_sh=False,
        )

        confidence = opacity_pkg["render"].mean(dim=0).clamp(0.0, 1.0)
        inv_depth = pkg["depth"].squeeze(0)
        depth = torch.where(
            inv_depth > 0,
            confidence / torch.clamp(inv_depth, min=1e-8),
            torch.zeros_like(inv_depth),
        )
        depth = torch.where(confidence >= opacity_threshold, depth, torch.zeros_like(depth))
    else:
        available = ", ".join(sorted(pkg.keys()))
        raise RuntimeError(
            f"Renderer output has no compatible depth for --depth_source {depth_source}. "
            f"Available keys: {available}"
        )

    if train_test_exp:
        depth = depth[..., depth.shape[-1] // 2:]
        image = image[..., image.shape[-1] // 2:]
        confidence = confidence[..., confidence.shape[-1] // 2:]

    if return_confidence:
        return depth, image, confidence
    return depth, image


def render_view_point_cloud(
    view,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    opacity_threshold,
    depth_source,
    rendered_depth_dir,
    rendered_color_dir,
    stride,
    min_depth,
    max_depth,
):
    depth, image = render_view_depth_color(
        view,
        gaussians,
        pipeline,
        background,
        train_test_exp,
        separate_sh,
        opacity_threshold,
        depth_source,
        rendered_depth_dir,
        rendered_color_dir,
    )
    return backproject_depth(view, depth, image, stride, min_depth, max_depth)


def collect_point_cloud(
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    opacity_threshold,
    depth_source,
    rendered_depth_dir,
    rendered_color_dir,
    stride,
    min_depth,
    max_depth,
    view_stride,
):
    xyz_chunks = []
    rgb_chunks = []

    selected_views = views[::view_stride]
    for view in tqdm(selected_views, desc="Extracting depth point cloud"):
        xyz, rgb = render_view_point_cloud(
            view,
            gaussians,
            pipeline,
            background,
            train_test_exp,
            separate_sh,
            opacity_threshold,
            depth_source,
            rendered_depth_dir,
            rendered_color_dir,
            stride,
            min_depth,
            max_depth,
        )
        if xyz is None:
            continue
        xyz_chunks.append(xyz)
        rgb_chunks.append((rgb * 255.0).astype(np.uint8))

    if not xyz_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    xyz = np.concatenate(xyz_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], rgb[finite]


def split_evaluate_args(extra_args):
    if len(extra_args) == 1 and extra_args[0] == "--":
        return []
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def run_evaluation(pred_path, gt_path, out_dir, evaluate_script, extra_args):
    cmd = [
        sys.executable,
        str(evaluate_script),
        "--pred",
        str(pred_path),
        "--gt",
        str(gt_path),
        "--out-dir",
        str(out_dir),
        *split_evaluate_args(extra_args),
    ]
    print("Running evaluation:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def camera_to_open3d_frame(view, depth, color, stride):
    fx = fov2focal(view.FoVx, view.image_width) / stride
    fy = fov2focal(view.FoVy, view.image_height) / stride
    cx = (view.image_width * 0.5) / stride
    cy = (view.image_height * 0.5) / stride
    world_to_camera = view.world_view_transform.detach().cpu().numpy().T
    camera_center = view.camera_center.detach().cpu().numpy()
    return {
        "width": int(depth.shape[1]),
        "height": int(depth.shape[0]),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "world_to_camera": world_to_camera.tolist(),
        "camera_center": camera_center.tolist(),
        "colmap_id": int(view.colmap_id),
        "image_name": view.image_name,
    }


def view_forward_from_world_to_camera(world_to_camera):
    rotation = world_to_camera[:3, :3]
    forward = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    norm = np.linalg.norm(forward)
    if norm <= 0:
        return forward
    return forward / norm


def make_geometric_pairs(frames, num_view):
    centers = np.asarray([frame["camera_center"] for frame in frames], dtype=np.float32)
    forwards = np.asarray([
        view_forward_from_world_to_camera(np.asarray(frame["world_to_camera"], dtype=np.float32))
        for frame in frames
    ], dtype=np.float32)
    pairs = {}
    for ref_idx in range(len(frames)):
        scored = []
        for src_idx in range(len(frames)):
            if src_idx == ref_idx:
                continue
            distance = float(np.linalg.norm(centers[src_idx] - centers[ref_idx]))
            facing = float(max(0.0, np.dot(forwards[ref_idx], forwards[src_idx])))
            score = facing / (distance + 1e-6)
            scored.append((score, src_idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        pairs[ref_idx] = [(idx, score) for score, idx in scored[:num_view]]
    return pairs


def read_colmap_image_tracks(source_path):
    sparse_dir = _find_colmap_sparse_dir(source_path)
    try:
        images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    except Exception:
        images = read_images_text(os.path.join(sparse_dir, "images.txt"))
    tracks_by_name = {}
    for image in images.values():
        image_name = Path(image.name).stem
        point_ids = np.asarray(image.point3D_ids)
        tracks_by_name[image_name] = set(int(pid) for pid in point_ids if int(pid) != -1)
    return tracks_by_name


def make_colmap_covisibility_pairs(frames, source_path, num_view):
    tracks_by_name = read_colmap_image_tracks(source_path)
    frame_tracks = []
    for frame in frames:
        name = Path(frame["image_name"]).stem
        frame_tracks.append(tracks_by_name.get(name, set()))
    if not any(frame_tracks):
        raise RuntimeError("No COLMAP tracks matched rendered frame names")

    pairs = {}
    for ref_idx, ref_tracks in enumerate(frame_tracks):
        scored = []
        for src_idx, src_tracks in enumerate(frame_tracks):
            if src_idx == ref_idx:
                continue
            common = len(ref_tracks & src_tracks)
            if common <= 0:
                continue
            scored.append((float(common), src_idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        pairs[ref_idx] = [(idx, score) for score, idx in scored[:num_view]]
    return pairs


def attach_mvs_pairs(frames, source_path, args):
    if args.mvs_pair_mode == "none":
        return frames, None
    try:
        if args.mvs_pair_mode == "colmap":
            pairs = make_colmap_covisibility_pairs(frames, source_path, args.mvs_pair_candidates)
            pair_source = "colmap_covisibility"
        elif args.mvs_pair_mode == "geometry":
            pairs = make_geometric_pairs(frames, args.mvs_pair_candidates)
            pair_source = "camera_geometry"
        else:
            try:
                pairs = make_colmap_covisibility_pairs(frames, source_path, args.mvs_pair_candidates)
                pair_source = "colmap_covisibility"
            except Exception as exc:
                print(f"Falling back to camera-geometry MVS pairs: {exc}")
                pairs = make_geometric_pairs(frames, args.mvs_pair_candidates)
                pair_source = "camera_geometry"
    except Exception as exc:
        if args.mvs_pair_mode in {"colmap", "geometry"}:
            raise
        print(f"Falling back to no metadata MVS pairs: {exc}")
        return frames, None

    for ref_idx, frame in enumerate(frames):
        frame["src_views"] = [int(idx) for idx, _ in pairs.get(ref_idx, [])]
        frame["src_view_scores"] = [float(score) for _, score in pairs.get(ref_idx, [])]
    return frames, pair_source


def write_pair_file(path, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(frames)}\n")
        for idx, frame in enumerate(frames):
            src_views = frame.get("src_views", [])
            scores = frame.get("src_view_scores", [0.0] * len(src_views))
            f.write(f"{idx}\n")
            entries = [str(len(src_views))]
            for src_idx, score in zip(src_views, scores):
                entries.extend([str(src_idx), f"{float(score):.6f}"])
            f.write(" ".join(entries) + "\n")


def render_fusion_frames(
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    args,
    frames_dir,
    rendered_depth_dir,
    rendered_color_dir,
    desc,
):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    selected_views = views[::args.view_stride]

    for frame_idx, view in enumerate(tqdm(selected_views, desc=desc)):
        depth, image, confidence = render_view_depth_color(
            view,
            gaussians,
            pipeline,
            background,
            train_test_exp,
            separate_sh,
            args.opacity_threshold,
            args.depth_source,
            rendered_depth_dir,
            rendered_color_dir,
            return_confidence=True,
        )
        if args.min_depth > 0:
            depth = torch.where(depth >= args.min_depth, depth, torch.zeros_like(depth))
        if args.max_depth > 0:
            depth = torch.where(depth <= args.max_depth, depth, torch.zeros_like(depth))

        if args.pixel_stride > 1:
            depth = depth[::args.pixel_stride, ::args.pixel_stride]
            image = image[:, ::args.pixel_stride, ::args.pixel_stride]
            confidence = confidence[::args.pixel_stride, ::args.pixel_stride]

        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        color_np = (image.permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
        valid = np.isfinite(depth_np) & (depth_np > 0)
        if valid.sum() == 0:
            continue

        depth_name = f"depth_{frame_idx:05d}.npy"
        color_name = f"color_{frame_idx:05d}.npy"
        confidence_name = f"confidence_{frame_idx:05d}.npy"
        np.save(frames_dir / depth_name, depth_np)
        np.save(frames_dir / color_name, color_np)
        np.save(frames_dir / confidence_name, confidence_np)

        frame = camera_to_open3d_frame(view, depth_np, color_np, args.pixel_stride)
        frame.update({
            "image_name": view.image_name,
            "depth": depth_name,
            "color": color_name,
            "confidence": confidence_name,
            "valid_depth_pixels": int(valid.sum()),
        })
        frames.append(frame)

    return frames


def render_tsdf_frames(
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    args,
    frames_dir,
    rendered_depth_dir,
    rendered_color_dir,
):
    return render_fusion_frames(
        views,
        gaussians,
        pipeline,
        background,
        train_test_exp,
        separate_sh,
        args,
        frames_dir,
        rendered_depth_dir,
        rendered_color_dir,
        "Rendering TSDF depth frames",
    )


def render_mvs_frames(
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    args,
    frames_dir,
    rendered_depth_dir,
    rendered_color_dir,
):
    return render_fusion_frames(
        views,
        gaussians,
        pipeline,
        background,
        train_test_exp,
        separate_sh,
        args,
        frames_dir,
        rendered_depth_dir,
        rendered_color_dir,
        "Rendering MVS depth frames",
    )


def run_tsdf_fusion(frames_dir, metadata_path, out_path, args):
    tsdf_helper = Path(args.tsdf_helper).expanduser()
    if not tsdf_helper.is_absolute():
        script_relative = Path(__file__).resolve().parent / tsdf_helper
        if script_relative.exists():
            tsdf_helper = script_relative
    if not Path(args.tsdf_python).exists():
        raise FileNotFoundError(f"--tsdf_python does not exist: {args.tsdf_python}")
    if not tsdf_helper.exists():
        raise FileNotFoundError(f"--tsdf_helper does not exist: {tsdf_helper}")
    cmd = [
        args.tsdf_python,
        str(tsdf_helper),
        "--frames-dir",
        str(frames_dir),
        "--metadata",
        str(metadata_path),
        "--out",
        str(out_path),
        "--voxel-length",
        str(args.tsdf_voxel_length),
        "--sdf-trunc",
        str(args.tsdf_sdf_trunc),
        "--depth-trunc",
        str(args.tsdf_depth_trunc),
        "--post-voxel-size",
        str(args.voxel_size),
        "--max-points",
        str(args.max_points),
        "--seed",
        str(args.seed),
    ]
    if args.tsdf_no_color:
        cmd.append("--no-color")
    if args.tsdf_remove_outliers:
        cmd.extend([
            "--remove-outliers",
            "--outlier-nb-neighbors",
            str(args.tsdf_outlier_nb_neighbors),
            "--outlier-std-ratio",
            str(args.tsdf_outlier_std_ratio),
        ])
    print("Running TSDF fusion:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_mvs_fusion(frames_dir, metadata_path, out_path, args):
    mvs_helper = Path(args.mvs_helper).expanduser()
    if not mvs_helper.is_absolute():
        script_relative = Path(__file__).resolve().parent / mvs_helper
        if script_relative.exists():
            mvs_helper = script_relative
    if not Path(args.mvs_python).exists():
        raise FileNotFoundError(f"--mvs_python does not exist: {args.mvs_python}")
    if not mvs_helper.exists():
        raise FileNotFoundError(f"--mvs_helper does not exist: {mvs_helper}")
    cmd = [
        args.mvs_python,
        str(mvs_helper),
        "--frames-dir",
        str(frames_dir),
        "--metadata",
        str(metadata_path),
        "--out",
        str(out_path),
        "--conf",
        str(args.mvs_conf),
        "--num-view",
        str(args.mvs_num_view),
        "--pair-mode",
        str(args.mvs_pair_use),
        "--post-voxel-size",
        str(args.voxel_size),
        "--max-points",
        str(args.max_points),
        "--seed",
        str(args.seed),
    ]
    if args.mvs_save_masks:
        cmd.append("--save-masks")
    print("Running MVS-style fusion:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def find_precomputed_frames(model_path, iteration, method, skip_train, skip_test):
    if method not in {"tsdf", "mvs"} or not skip_test:
        return None

    model_path = Path(model_path).expanduser()
    if model_path.name.endswith("_geosvr_depth"):
        out_dir = model_path
    else:
        suffixes = []
        if iteration > 0:
            suffixes.append(f"ours_{iteration}_geosvr_depth")
        suffixes.append("ours_20000_geosvr_depth")
        out_dir = None
        for suffix in suffixes:
            candidate = model_path / "train" / suffix
            if (candidate / "train_frames_meta.json").is_file():
                out_dir = candidate
                break
        if out_dir is None:
            return None

    frames_dir = out_dir / "train_frames"
    metadata_path = out_dir / "train_frames_meta.json"
    if not frames_dir.is_dir() or not metadata_path.is_file():
        return None
    return "train", out_dir, frames_dir, metadata_path


def run_precomputed_frames_fusion(args):
    precomputed = find_precomputed_frames(
        args.model_path,
        args.iteration,
        args.method,
        args.skip_train,
        args.skip_test,
    )
    if precomputed is None:
        return False

    set_name, out_dir, frames_dir, metadata_path = precomputed
    ply_path = out_dir / (args.tsdf_ply_name if args.method == "tsdf" else args.mvs_ply_name)
    if args.method == "tsdf":
        run_tsdf_fusion(frames_dir, metadata_path, ply_path, args)
    else:
        run_mvs_fusion(frames_dir, metadata_path, ply_path, args)

    with metadata_path.open("r", encoding="utf-8") as f:
        num_views = len(json.load(f).get("frames", []))
    meta = {
        "set": set_name,
        "iteration": args.iteration,
        "method": args.method,
        "num_views": num_views,
        "pixel_stride": args.pixel_stride,
        "view_stride": args.view_stride,
        "conf_threshold": args.mvs_conf if args.method == "mvs" else None,
        "voxel_size": args.voxel_size,
        "max_points": args.max_points,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "num_points_before_downsample": None,
        "num_points": None,
        "depth_source": args.depth_source,
        "frames_dir": str(frames_dir),
        "frames_meta": str(metadata_path),
    }
    with (out_dir / "depth_point_cloud_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote point cloud to {ply_path}")
    return True


def parse_cli_args(parser):
    args_cmdline = parser.parse_args(sys.argv[1:])
    cfg_path = None
    if args_cmdline.model_path:
        cfg_path = Path(args_cmdline.model_path) / "cfg_args"
    if cfg_path is None or cfg_path.is_file():
        return get_combined_args(parser)

    print("Looking for config file in", cfg_path)
    precomputed = find_precomputed_frames(
        args_cmdline.model_path,
        args_cmdline.iteration,
        args_cmdline.method,
        args_cmdline.skip_train,
        args_cmdline.skip_test,
    )
    if precomputed is None:
        raise FileNotFoundError(f"No cfg_args or precomputed train_frames_meta.json found for --model_path: {args_cmdline.model_path}")
    print(f"Config file not found; using precomputed frames metadata: {precomputed[3]}")
    return Namespace(**vars(args_cmdline))


def extract_sets(
    dataset,
    iteration,
    pipeline,
    skip_train,
    skip_test,
    separate_sh,
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        output_root = Path(args.output_path or dataset.model_path)
        set_views = []
        if not skip_train:
            set_views.append(("train", scene.getTrainCameras()))
        if not skip_test:
            set_views.append(("test", scene.getTestCameras()))

        written = []
        for name, views in set_views:
            out_dir = output_root / name / f"ours_{scene.loaded_iter}"
            makedirs(out_dir, exist_ok=True)
            rendered_depth_dir = infer_rendered_depth_dir(output_root, name, scene.loaded_iter, args)
            rendered_color_dir = infer_rendered_color_dir(output_root, name, scene.loaded_iter, args)
            if args.depth_source == "precomputed" and rendered_depth_dir is None:
                raise FileNotFoundError(
                    f"--depth_source precomputed requires --rendered_depth_root or an existing "
                    f"{out_dir / 'renders_depth'} directory"
                )
            if rendered_depth_dir is not None:
                print(f"Using rendered depth directory for {name}: {rendered_depth_dir}")
            if rendered_color_dir is not None:
                print(f"Using rendered color directory for {name}: {rendered_color_dir}")
            if not views:
                print(f"Skipping {name}: no cameras found")
                continue
            if args.method in {"tsdf", "mvs"}:
                frames_dir = out_dir / ("tsdf_frames" if args.method == "tsdf" else "mvs_frames")
                render_frames = render_tsdf_frames if args.method == "tsdf" else render_mvs_frames
                frames = render_frames(
                    views,
                    gaussians,
                    pipeline,
                    background,
                    dataset.train_test_exp,
                    separate_sh,
                    args,
                    frames_dir,
                    rendered_depth_dir,
                    rendered_color_dir,
                )
                if not frames:
                    raise RuntimeError(
                        f"No valid {args.method.upper()} frames were rendered for {name}. "
                        "Try lowering --opacity_threshold, increasing --max_depth, "
                        "or checking the camera/model paths."
                    )
                pair_source = None
                if args.method == "mvs":
                    frames, pair_source = attach_mvs_pairs(frames, dataset.source_path, args)
                    if pair_source is not None:
                        write_pair_file(out_dir / "mvs_pair.txt", frames)
                ply_path = out_dir / (args.tsdf_ply_name if args.method == "tsdf" else args.mvs_ply_name)
                fusion_meta_path = out_dir / ("tsdf_frames_meta.json" if args.method == "tsdf" else "mvs_frames_meta.json")
                with fusion_meta_path.open("w", encoding="utf-8") as f:
                    json.dump({"frames": frames, "pair_source": pair_source}, f, indent=2)
                if args.method == "tsdf":
                    run_tsdf_fusion(frames_dir, fusion_meta_path, ply_path, args)
                else:
                    run_mvs_fusion(frames_dir, fusion_meta_path, ply_path, args)
                before_downsample = None
                num_points = None
            else:
                xyz, rgb = collect_point_cloud(
                    views,
                    gaussians,
                    pipeline,
                    background,
                    dataset.train_test_exp,
                    separate_sh,
                    args.opacity_threshold,
                    args.depth_source,
                    rendered_depth_dir,
                    rendered_color_dir,
                    args.pixel_stride,
                    args.min_depth,
                    args.max_depth,
                    args.view_stride,
                )
                before_downsample = len(xyz)
                xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size)
                xyz, rgb = random_subsample(xyz, rgb, args.max_points, args.seed)

                ply_path = out_dir / args.ply_name
                write_ply(ply_path, xyz, rgb)
                num_points = int(len(xyz))
            written.append(ply_path)

            meta = {
                "set": name,
                "iteration": scene.loaded_iter,
                "num_views": len(views[::args.view_stride]),
                "method": args.method,
                "num_points_before_downsample": None if before_downsample is None else int(before_downsample),
                "num_points": num_points,
                "pixel_stride": args.pixel_stride,
                "view_stride": args.view_stride,
                "opacity_threshold": args.opacity_threshold,
                "depth_source": args.depth_source,
                "rendered_depth_dir": None if rendered_depth_dir is None else str(rendered_depth_dir),
                "rendered_color_dir": None if rendered_color_dir is None else str(rendered_color_dir),
                "voxel_size": args.voxel_size,
                "min_depth": args.min_depth,
                "max_depth": args.max_depth,
                "tsdf_voxel_length": args.tsdf_voxel_length if args.method == "tsdf" else None,
                "tsdf_sdf_trunc": args.tsdf_sdf_trunc if args.method == "tsdf" else None,
                "tsdf_depth_trunc": args.tsdf_depth_trunc if args.method == "tsdf" else None,
                "mvs_conf": args.mvs_conf if args.method == "mvs" else None,
                "mvs_num_view": args.mvs_num_view if args.method == "mvs" else None,
                "mvs_pair_mode": args.mvs_pair_mode if args.method == "mvs" else None,
                "mvs_pair_use": args.mvs_pair_use if args.method == "mvs" else None,
            }
            with (out_dir / "depth_point_cloud_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"Wrote point cloud to {ply_path}")

        if args.gt:
            if not written:
                raise RuntimeError("No point cloud was written, cannot run evaluation")
            eval_pred = Path(args.eval_pred) if args.eval_pred else written[-1]
            eval_out = Path(args.eval_out_dir or (output_root / "reconstruction_eval"))
            run_evaluation(
                eval_pred,
                Path(args.gt),
                eval_out,
                Path(args.evaluate_script),
                args.evaluate_args,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract a dense point cloud from rendered 3DGS depth maps")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--method", choices=["direct", "tsdf", "mvs"], default="direct")
    parser.add_argument("--ply_name", default="depth_point_cloud.ply", type=str)
    parser.add_argument("--tsdf_ply_name", default="tsdf_point_cloud.ply", type=str)
    parser.add_argument("--mvs_ply_name", default="mvs_point_cloud.ply", type=str)
    parser.add_argument("--pixel_stride", default=2, type=int)
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--opacity_threshold", default=0.5, type=float)
    parser.add_argument(
        "--depth_source",
        choices=["auto", "3dgs", "inverse", "planargs", "planargs_plane", "plane", "precomputed"],
        default="auto",
        type=str,
        help="Depth source: auto uses precomputed renders_depth when present, then PlanarGS plane_depth, then 3DGS inverse depth.",
    )
    parser.add_argument(
        "--rendered_depth_root",
        default="",
        type=str,
        help="Directory containing precomputed per-view depth files, e.g. PlanarGS train/ours_30000/renders_depth.",
    )
    parser.add_argument(
        "--rendered_color_root",
        default="",
        type=str,
        help="Directory containing precomputed rendered colors, e.g. PlanarGS train/ours_30000/renders.",
    )
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--max_points", default=0, type=int)
    parser.add_argument("--min_depth", default=0.0, type=float)
    parser.add_argument("--max_depth", default=0.0, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--gt", default="", type=str, help="Optional GT PLY for LingBot-Map evaluation")
    parser.add_argument("--eval_pred", default="", type=str, help="Optional extracted PLY to evaluate")
    parser.add_argument("--eval_out_dir", default="", type=str)
    parser.add_argument("--evaluate_script", default=str(Path(__file__).resolve().parent / "evaluate_reconstruction.py"), type=str)
    parser.add_argument("evaluate_args", nargs="*", help="Arguments forwarded to evaluate_reconstruction.py; prefix with --")
    parser.add_argument("--tsdf_python", default=sys.executable, type=str)
    parser.add_argument("--tsdf_helper", default="tsdf_fuse_point_cloud.py", type=str)
    parser.add_argument("--tsdf_voxel_length", default=0.02, type=float)
    parser.add_argument("--tsdf_sdf_trunc", default=0.08, type=float)
    parser.add_argument("--tsdf_depth_trunc", default=8.0, type=float)
    parser.add_argument("--tsdf_no_color", action="store_true")
    parser.add_argument("--tsdf_remove_outliers", action="store_true")
    parser.add_argument("--tsdf_outlier_nb_neighbors", default=30, type=int)
    parser.add_argument("--tsdf_outlier_std_ratio", default=2.0, type=float)
    parser.add_argument("--mvs_python", default=sys.executable, type=str)
    parser.add_argument("--mvs_helper", default="mvs_fuse_point_cloud.py", type=str)
    parser.add_argument("--mvs_conf", default=0.5, type=float)
    parser.add_argument("--mvs_num_view", default=10, type=int)
    parser.add_argument("--mvs_pair_mode", choices=["auto", "colmap", "geometry", "none"], default="auto")
    parser.add_argument("--mvs_pair_use", choices=["metadata", "geometry"], default="metadata")
    parser.add_argument("--mvs_pair_candidates", default=50, type=int)
    parser.add_argument("--mvs_save_masks", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parse_cli_args(parser)

    if args.pixel_stride < 1:
        raise ValueError("--pixel_stride must be >= 1")
    if args.view_stride < 1:
        raise ValueError("--view_stride must be >= 1")
    if args.mvs_num_view < 1:
        raise ValueError("--mvs_num_view must be >= 1")
    if args.mvs_pair_candidates < args.mvs_num_view:
        args.mvs_pair_candidates = args.mvs_num_view
    if args.skip_train and args.skip_test:
        raise ValueError("Both --skip_train and --skip_test were set; nothing to extract")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"--model_path does not exist: {args.model_path}")
    if run_precomputed_frames_fusion(args):
        sys.exit(0)
    if not Path(args.source_path).exists():
        raise FileNotFoundError(f"--source_path does not exist: {args.source_path}")

    print("Extracting depth point cloud from " + args.model_path)
    safe_state(args.quiet)
    extract_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        args,
    )
