import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def random_subsample(points, colors, max_points, seed):
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=max_points, replace=False)
    return points[keep], colors[keep] if colors is not None else None


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse rendered RGB-D frames with Open3D TSDF and output a point cloud")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel-length", type=float, default=0.02)
    parser.add_argument("--sdf-trunc", type=float, default=0.08)
    parser.add_argument("--depth-trunc", type=float, default=8.0)
    parser.add_argument("--post-voxel-size", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=30)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.metadata.open("r", encoding="utf-8") as f:
        frames = json.load(f)["frames"]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=(
            o3d.pipelines.integration.TSDFVolumeColorType.NoColor
            if args.no_color
            else o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        ),
    )

    for frame in frames:
        depth = np.load(args.frames_dir / frame["depth"]).astype(np.float32)
        color = np.load(args.frames_dir / frame["color"]).astype(np.uint8)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            frame["width"],
            frame["height"],
            frame["fx"],
            frame["fy"],
            frame["cx"],
            frame["cy"],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.asarray(frame["world_to_camera"], dtype=np.float64))

    pcd = volume.extract_point_cloud()
    if args.remove_outliers and len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )
    if args.post_voxel_size > 0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(args.post_voxel_size)
    if args.max_points > 0 and len(pcd.points) > args.max_points:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        points, colors = random_subsample(points, colors, args.max_points, args.seed)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out), pcd, write_ascii=False, compressed=False)
    print(f"Wrote {len(pcd.points):,} TSDF points to {args.out}")


if __name__ == "__main__":
    main()
