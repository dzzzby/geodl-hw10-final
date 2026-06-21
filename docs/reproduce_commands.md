# 复现实验命令

以下命令是本项目的复现入口。第三方仓库建议放在 `third_party/` 下，数据放在 `data/Sequence_01` 至 `data/Sequence_05`。

## 1. GeoSVR 主流程

GeoSVR 用于 Sequence_01、Sequence_02、Sequence_05，并作为 Sequence_03/04 的对比方法。

```bash
git clone https://github.com/Fictionarry/GeoSVR third_party/GeoSVR
cd third_party/GeoSVR
```

按 GeoSVR 官方 README 配置环境后，对每个序列训练 3DGS 模型。训练完成后，使用本仓库脚本导出 MVS 点云：

```bash
python ../../scripts/extract_depth_point_cloud.py \
  -m output/Sequence_01 \
  --skip_test \
  --pixel_stride 2 \
  --voxel_size 0.01
```

然后评估：

```bash
python ../../scripts/evaluate_reconstruction.py \
  --pred output/Sequence_01/train/ours_30000/depth_point_cloud.ply \
  --gt ../../data/Sequence_01/gt/gt_pd.ply \
  --out-dir ../../eval_outputs/geosvr/Sequence_01 \
  --sample-align 100000 \
  --sample-eval 0 \
  --icp-iterations 300 \
  --pca-permutations
```

## 2. PlanarGS 对比流程

PlanarGS 用于平面先验对比，其中 Sequence_03 采用该方法作为最终结果。

```bash
git clone https://github.com/SJTU-ViSYS-team/PlanarGS third_party/PlanarGS
cd third_party/PlanarGS
```

训练完成后同样导出点云并评估：

```bash
python ../../scripts/evaluate_reconstruction.py \
  --pred output/Sequence_03/mvs_point_cloud_planargs.ply \
  --gt ../../data/Sequence_03/gt/gt_pd.ply \
  --out-dir ../../eval_outputs/planargs/Sequence_03 \
  --sample-align 100000 \
  --sample-eval 0 \
  --icp-iterations 300 \
  --pca-permutations
```

## 3. LingBot-Map 特殊序列流程

Sequence_04 的主要问题是 COLMAP 位姿不稳定，因此使用 LingBot-Map。

```bash
git clone https://github.com/robbyant/lingbot-map third_party/lingbot-map
cd third_party/lingbot-map
python demo.py \
  --model_path checkpoints/lingbot-map/lingbot-map/lingbot-map-long.pt \
  --image_folder ../../data/Sequence_04/images \
  --mask_sky
```

LingBot-Map 输出 `export.glb` 后可直接评估：

```bash
python ../../scripts/evaluate_reconstruction.py \
  --pred export.glb \
  --gt ../../data/Sequence_04/gt/gt_pd.ply \
  --out-dir ../../eval_outputs/lingbot/Sequence_04 \
  --sample-align 300000 \
  --sample-eval 0 \
  --icp-iterations 300 \
  --pca-permutations
```

## 4. 生成可视化叠加点云

```bash
python scripts/make_pointcloud_overlay.py \
  --pred eval_outputs/lingbot/Sequence_04/pred_aligned.ply \
  --gt data/Sequence_04/gt/gt_pd.ply \
  --out eval_outputs/lingbot/Sequence_04/overlay_pred_red_gt_blue.ply
```

红色表示预测点云，蓝色表示 GT 点云。该文件用于人工检查配准是否正确，以及定位缺失/错位区域。
