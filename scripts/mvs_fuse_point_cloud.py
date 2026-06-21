import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCALE_START = 1
SCALE_END = 11
DIST_BASE = 1.0 / 2.0
DIFF_BASE = 0.25


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


def save_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


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


def intrinsics_from_frame(frame):
    return np.array([
        [frame["fx"], 0.0, frame["cx"]],
        [0.0, frame["fy"], frame["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def load_frame(frames_dir, frame):
    depth = np.load(frames_dir / frame["depth"]).astype(np.float32)
    color = np.load(frames_dir / frame["color"]).astype(np.uint8)
    confidence_name = frame.get("confidence")
    if confidence_name:
        confidence = np.load(frames_dir / confidence_name).astype(np.float32)
    else:
        confidence = (depth > 0).astype(np.float32)
    return {
        "depth": depth,
        "color": color,
        "confidence": confidence,
        "intrinsics": intrinsics_from_frame(frame),
        "extrinsics": np.asarray(frame["world_to_camera"], dtype=np.float32),
    }


def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]

    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]),
    )

    xyz_src = np.matmul(
        np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
        np.vstack((xyz_ref, np.ones_like(x_ref))),
    )[:3]
    k_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = k_xyz_src[:2] / np.clip(k_xyz_src[2:3], 1e-8, None)

    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(
        depth_src,
        x_src,
        y_src,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]),
    )
    xyz_reprojected = np.matmul(
        np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
        np.vstack((xyz_src, np.ones_like(x_ref))),
    )[:3]

    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    k_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = k_xyz_reprojected[:2] / np.clip(k_xyz_reprojected[2:3], 1e-8, None)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x_reprojected, y_reprojected = reproject_with_depth(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src,
    )

    dist = np.sqrt((x_reprojected - x_ref) ** 2 + (y_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    valid_depth = (depth_ref > 0) & (depth_src.max() > 0) & (depth_reprojected > 0)

    masks = []
    mask = None
    for i in range(SCALE_START, SCALE_END):
        mask = np.logical_and(dist < i * DIST_BASE, depth_diff < math.log(max(i, 1.05), 10) * DIFF_BASE)
        mask = np.logical_and(mask, valid_depth)
        masks.append(mask)

    depth_reprojected[~mask] = 0
    return masks, mask, depth_reprojected


def select_source_views(frames, ref_idx, max_sources):
    ref_center = camera_center_from_frame(frames[ref_idx])
    distances = []
    for idx, frame in enumerate(frames):
        if idx == ref_idx:
            continue
        center = camera_center_from_frame(frame)
        distances.append((float(np.linalg.norm(center - ref_center)), idx))
    distances.sort(key=lambda item: item[0])
    return [idx for _, idx in distances[:max_sources]]


def select_metadata_source_views(frames, ref_idx, max_sources):
    src_views = frames[ref_idx].get("src_views", [])
    src_views = [int(idx) for idx in src_views if int(idx) != ref_idx and 0 <= int(idx) < len(frames)]
    if max_sources > 0:
        src_views = src_views[:max_sources]
    return src_views


def camera_center_from_frame(frame):
    if "camera_center" in frame:
        return np.asarray(frame["camera_center"], dtype=np.float32)
    world_to_camera = np.asarray(frame["world_to_camera"], dtype=np.float32)
    return np.linalg.inv(world_to_camera)[:3, 3].astype(np.float32)


def frame_points(ref, src_frames, conf_threshold, save_masks_dir=None, frame_idx=0):
    ref_depth = ref["depth"]
    ref_confidence = ref["confidence"]
    if ref_confidence.shape != ref_depth.shape:
        ref_confidence = cv2.resize(ref_confidence, (ref_depth.shape[1], ref_depth.shape[0]))

    photo_mask = np.logical_and(ref_confidence > conf_threshold, ref_depth > 0)
    geo_mask_sum = np.zeros_like(ref_depth, dtype=np.int32)
    geo_mask_sums = [np.zeros_like(ref_depth, dtype=np.int32) for _ in range(SCALE_END - SCALE_START)]
    depth_reprojected_sum = np.zeros_like(ref_depth, dtype=np.float32)

    for src in src_frames:
        masks, geo_mask, depth_reprojected = check_geometric_consistency(
            ref_depth,
            ref["intrinsics"],
            ref["extrinsics"],
            src["depth"],
            src["intrinsics"],
            src["extrinsics"],
        )
        geo_mask_sum += geo_mask.astype(np.int32)
        depth_reprojected_sum += depth_reprojected
        for i, mask in enumerate(masks):
            geo_mask_sums[i] += mask.astype(np.int32)

    depth_est_averaged = (depth_reprojected_sum + ref_depth) / (geo_mask_sum + 1)
    depth_est_averaged[ref_confidence > 0.75] = ref_depth[ref_confidence > 0.75]

    geo_mask = geo_mask_sum >= SCALE_END
    for i in range(SCALE_START, SCALE_END):
        geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - SCALE_START] >= i)

    final_mask = np.logical_and(photo_mask, geo_mask)
    if save_masks_dir is not None:
        save_mask(save_masks_dir / f"{frame_idx:08d}_photo.png", photo_mask)
        save_mask(save_masks_dir / f"{frame_idx:08d}_geo.png", geo_mask)
        save_mask(save_masks_dir / f"{frame_idx:08d}_final.png", final_mask)

    height, width = depth_est_averaged.shape
    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x, y, depth = x[final_mask], y[final_mask], depth_est_averaged[final_mask]
    if len(depth) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), final_mask.mean()

    xyz_ref = np.matmul(
        np.linalg.inv(ref["intrinsics"]),
        np.vstack((x, y, np.ones_like(x))) * depth,
    )
    xyz_world = np.matmul(
        np.linalg.inv(ref["extrinsics"]),
        np.vstack((xyz_ref, np.ones_like(x))),
    )[:3]
    colors = ref["color"][final_mask]
    return xyz_world.transpose((1, 0)).astype(np.float32), colors.astype(np.uint8), final_mask.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse rendered RGB-D frames with EC-MVSNet-style depth filtering")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--num-view", type=int, default=10)
    parser.add_argument("--pair-mode", choices=["metadata", "geometry"], default="metadata")
    parser.add_argument("--post-voxel-size", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-masks", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.metadata.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    frames = metadata["frames"]

    loaded = [load_frame(args.frames_dir, frame) for frame in frames]
    for frame, data in zip(frames, loaded):
        data["camera_center"] = camera_center_from_frame(frame)

    xyz_chunks = []
    rgb_chunks = []
    mask_dir = args.out.parent / "mvs_masks" if args.save_masks else None
    for ref_idx, ref in enumerate(loaded):
        if args.pair_mode == "metadata" and frames[ref_idx].get("src_views"):
            src_indices = select_metadata_source_views(frames, ref_idx, args.num_view)
        else:
            src_indices = select_source_views(frames, ref_idx, args.num_view)
        src_frames = [loaded[idx] for idx in src_indices]
        xyz, rgb, valid_ratio = frame_points(
            ref,
            src_frames,
            args.conf,
            save_masks_dir=mask_dir,
            frame_idx=ref_idx,
        )
        print(f"processing ref-view{ref_idx:0>2}, final-mask:{valid_ratio}")
        if len(xyz) == 0:
            continue
        xyz_chunks.append(xyz)
        rgb_chunks.append(rgb)

    if xyz_chunks:
        xyz = np.concatenate(xyz_chunks, axis=0)
        rgb = np.concatenate(rgb_chunks, axis=0)
        finite = np.isfinite(xyz).all(axis=1)
        xyz, rgb = xyz[finite], rgb[finite]
    else:
        xyz = np.empty((0, 3), dtype=np.float32)
        rgb = np.empty((0, 3), dtype=np.uint8)

    xyz, rgb = voxel_downsample(xyz, rgb, args.post_voxel_size)
    xyz, rgb = random_subsample(xyz, rgb, args.max_points, args.seed)
    write_ply(args.out, xyz, rgb)
    print(f"Wrote {len(xyz):,} MVS-filtered points to {args.out}")


if __name__ == "__main__":
    main()
