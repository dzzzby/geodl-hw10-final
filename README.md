# GeoDL HW10 Final Submission

本目录按 GitHub 仓库格式整理了 HW10 Stage 2 的技术报告、评估脚本、指标表和第三方方法说明。

## 方法概述

本项目综合调研并调参了三条室内 3DGS 重建路线：

- GeoSVR：深度估计与几何约束更稳定，作为主要方法。
- PlanarGS：利用室内平面先验，在 Sequence_03 上表现最好。
- LingBot-Map：用于 Sequence_04 这类 COLMAP 位姿困难序列，最终取得该序列最佳结果。

最终平均 PSNR 记录为 `26.88 dB`。几何评估以预测点云和 GT 点云的双向最近邻距离为准，统计 10cm/20cm 阈值下的覆盖率与 F-score。

## 目录结构

```text
.
├── README.md
├── docs/
│   ├── technical_report.md
│   ├── reproduce_commands.md
│   ├── submission_checklist.md
│   └── members.md
├── metrics/
│   └── summary.md
├── scripts/
│   ├── evaluate_reconstruction.py
│   ├── extract_depth_point_cloud.py
│   ├── make_pointcloud_overlay.py
│   ├── mvs_fuse_point_cloud.py
│   └── tsdf_fuse_point_cloud.py
├── third_party/
│   └── THIRD_PARTY.md
└── assets/
    └── stage1_colmap.png
```

## 环境依赖

评估脚本依赖：

```bash
pip install numpy scipy trimesh open3d opencv-python pillow tqdm torch
```

其中 `extract_depth_point_cloud.py` 需要在 Gaussian Splatting/GeoSVR/PlanarGS 代码环境中运行，因为它会调用对应项目的 `scene`、`arguments`、`gaussian_renderer` 等模块。

## 复现流程

1. 按 `third_party/THIRD_PARTY.md` 拉取 GeoSVR、PlanarGS、LingBot-Map 三个第三方仓库。
2. 将共享数据集放到 `data/Sequence_01` 至 `data/Sequence_05`，每个序列包含 `images/` 和 `gt/gt_pd.ply`。
3. 分别运行 GeoSVR、PlanarGS 或 LingBot-Map，得到训练输出或导出点云。
4. 使用 `scripts/extract_depth_point_cloud.py`、`scripts/mvs_fuse_point_cloud.py` 或 `scripts/tsdf_fuse_point_cloud.py` 导出可评估点云。
5. 使用 `scripts/evaluate_reconstruction.py` 计算指标。

示例：

```bash
python scripts/evaluate_reconstruction.py \
  --pred data/Sequence_01/mvs_point_cloud_geosvr.ply \
  --gt data/Sequence_01/gt/gt_pd.ply \
  --out-dir eval_outputs/Sequence_01_geosvr_mvs \
  --sample-align 100000 \
  --sample-eval 0 \
  --icp-iterations 300 \
  --pca-permutations
```

## 最终结果

| Sequence | 最终方法 | F@10cm | F@20cm |
|---|---|---:|---:|
| Sequence_01 | GeoSVR + MVS | 0.991179 | 0.999513 |
| Sequence_02 | GeoSVR + MVS | 0.905278 | 0.976142 |
| Sequence_03 | PlanarGS + MVS | 0.984427 | 0.992249 |
| Sequence_04 | LingBot-Map | 0.738589 | 0.852390 |
| Sequence_05 | GeoSVR + MVS | 0.930279 | 0.963214 |

完整对比表见 `metrics/summary.md`，完整技术说明见 `docs/technical_report.md`。

复现实验入口见 `docs/reproduce_commands.md`。
