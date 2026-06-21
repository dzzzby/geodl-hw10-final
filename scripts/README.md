# Evaluation Scripts

## `evaluate_reconstruction.py`

统一点云精度评估脚本。支持 `.ply`、LingBot-Map 导出的 `.glb`，也支持将 mesh 表面采样后评估。

主要输出：

- `pred_extracted.ply`：从输入文件提取的原始预测点云。
- `pred_to_gt_transform.txt`：预测点云到 GT 坐标系的 4x4 Sim(3) 变换矩阵。
- `pred_aligned.ply`：对齐后的预测点云。
- `metrics.json`：双向距离、Chamfer、10cm/20cm 阈值内比例与 F-score。
- `pred_eval_colored_by_error.ply`：预测点云误差着色。
- `gt_eval_colored_by_error.ply`：GT 覆盖误差着色。

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

## `extract_depth_point_cloud.py`

在 3DGS/GeoSVR/PlanarGS 训练输出目录中重新渲染深度，并将深度图反投影为点云。该脚本需要放在对应 3DGS 类项目环境中运行。

示例：

```bash
python scripts/extract_depth_point_cloud.py \
  -m output/Sequence_01 \
  --skip_test \
  --pixel_stride 2 \
  --voxel_size 0.01
```

## `mvs_fuse_point_cloud.py`

对多帧深度点云执行多视图一致性过滤，输出更适合评估的 MVS 点云。

## `tsdf_fuse_point_cloud.py`

使用 Open3D TSDF 融合多帧 RGB-D，输出连续表面点云。

## `make_pointcloud_overlay.py`

生成红蓝双色叠加点云，用于人工检查预测点云与 GT 的配准质量。

示例：

```bash
python scripts/make_pointcloud_overlay.py \
  --pred eval_outputs/Sequence_04_lingbot/pred_aligned.ply \
  --gt data/Sequence_04/gt/gt_pd.ply \
  --out eval_outputs/Sequence_04_lingbot/overlay_pred_red_gt_blue.ply
```
