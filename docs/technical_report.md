# 3D Gaussian Splatting 室内场景重建技术报告

姓名：朱博怿  
任务：GeoDL HW10 Stage 2 共享室内场景重建  
核心方法：GeoSVR / PlanarGS / LingBot-Map 多方法调研、调参、评估与择优融合

## 1. Introduction

本次作业面向旧改施工中的室内三维重建问题：输入普通相机或手机采集的室内多视角图像，输出可用于浏览、测量和施工勘测的三维场景模型。该任务的难点主要来自室内场景纹理重复、局部弱纹理、遮挡多、墙面/地面等大平面占比高，以及相机轨迹在狭窄空间内容易产生位姿漂移。

课程给出的基础方案是 COLMAP + 3D Gaussian Splatting。COLMAP 负责从图像恢复相机位姿和稀疏点云，3DGS 使用可微渲染优化三维高斯的位置、尺度、颜色与不透明度，从而生成高质量新视角渲染结果。但在共享室内数据中，单纯依赖 COLMAP 的位姿质量并不总是稳定；一旦某些序列位姿漂移或局部失败，后续高斯优化会被错误几何约束放大，表现为墙体错位、漂浮物、尺度不一致和覆盖缺失。

因此，本项目采用“多论文调研 + 多代码库调参 + 统一评估择优”的策略。具体调研和尝试了 GeoSVR、PlanarGS 与 LingBot-Map 三条技术路线，并基于自写评估脚本对预测点云和 GT 点云进行双向精度评估。最终在大多数序列上采用 GeoSVR，在 Sequence_04 上采用不强依赖 COLMAP 的 LingBot-Map。

## 2. Method

### 2.1 总体流程

```text
共享图像序列
    |
    |-- GeoSVR: 深度估计 + 3DGS 训练 + MVS/TSDF 点云导出
    |
    |-- PlanarGS: 平面先验约束 + 3DGS 训练 + MVS/TSDF 点云导出
    |
    |-- LingBot-Map: 无 COLMAP/弱 COLMAP 依赖的地图重建，主要用于位姿困难序列
    |
统一评估脚本
    |-- 点云/GLB 读取
    |-- PCA + trimmed similarity ICP 自动对齐
    |-- 双向最近邻距离
    |-- 10cm/20cm 阈值内比例与 F-score
    |
逐序列选择最优结果并汇总
```

### 2.2 技术选型

**GeoSVR**  
GeoSVR 的优势在于引入更可靠的深度估计与几何约束。室内场景中墙、地面、家具边界往往存在大面积低纹理区域，COLMAP 稀疏点云在这些位置覆盖不足。GeoSVR 的深度引导能够为高斯优化提供更密集的几何信息，减少空洞和局部尺度漂移。本项目中 Sequence_01、Sequence_02、Sequence_05 的最佳结果均来自 `geosvr+mvs`，Sequence_03 中 GeoSVR 与 PlanarGS 表现接近。

**PlanarGS**  
PlanarGS 面向室内场景的大平面结构，引入平面先验以改善墙面、地面、天花板等区域的几何一致性。它在 Sequence_03 上取得了最好的 F-score，说明平面结构强、位姿质量较好时，平面先验可以显著提升完整度和几何规整性。但在 Sequence_02 和 Sequence_04 中，PlanarGS 对位姿和初始化质量仍较敏感。

**LingBot-Map**  
Sequence_04 是最明显的失败案例：GeoSVR 与 PlanarGS 在该序列上的 20cm F-score 最高也只有约 0.61，说明主要瓶颈不是后端高斯表达，而是位姿/地图初始化质量。LingBot-Map 不采用传统 COLMAP 流程作为核心前置，因此在该序列中显著改善全局几何一致性，20cm F-score 达到 0.852390，是 Sequence_04 的最终选择。

### 2.3 点云导出与融合

为了从不同方法得到可比较的几何结果，本项目使用两种点云导出方式：

1. MVS 点云：从渲染深度或多视图一致性深度反投影得到稠密点云，并使用多视图重投影一致性过滤噪声。
2. TSDF 点云：将多帧 RGB-D 渲染结果融合进 TSDF 体，适合获得更连续的表面，但可能在薄结构或错位位姿下产生过度平滑。

实验表明，在本数据集上 MVS 点云通常优于 TSDF 点云。原因是 TSDF 融合对相机位姿和深度尺度误差更敏感，位姿轻微偏差会在融合体中形成模糊厚墙或重影；MVS 点云虽然不如 TSDF 连续，但更能保留局部准确结构。

### 2.4 精度评估工具

本项目的创新点主要体现在自写的点云精度评估代码，位于 `scripts/evaluate_reconstruction.py`。它能够对不同来源的重建结果进行统一评估，支持：

- 输入格式：PLY 点云、LingBot-Map 导出的 GLB 点云，以及三角网格表面采样。
- 自动对齐：基于 PCA 主方向生成初始候选，再使用 trimmed similarity ICP 估计旋转、平移和统一尺度。
- 双向评估：同时计算 `pred -> gt` 和 `gt -> pred` 最近邻距离，避免只看单向距离导致“预测点少但看似很准”的偏差。
- 阈值指标：统计 10cm 与 20cm 阈值内比例，并计算 F-score。
- 可视化输出：导出对齐后的预测点云、误差着色点云和红蓝双色叠加点云，便于定位失败区域。

核心指标定义如下：

```text
Precision@t = Recon->GT within@t
Recall@t    = GT->Recon within@t
F@t         = 2 * Precision@t * Recall@t / (Precision@t + Recall@t)
```

其中 `GT->Recon` 衡量 GT 区域是否被完整覆盖，`Recon->GT` 衡量重建结果是否偏离真实结构。

## 3. Experiments

### 3.1 实验设置

本项目在共享室内数据集的五个序列上进行实验。对 GeoSVR、PlanarGS 分别尝试 MVS 与 TSDF 点云导出；对于位姿困难的 Sequence_04，额外使用 LingBot-Map。渲染质量以 PSNR/SSIM 作为参考，几何质量以 10cm/20cm 阈值下的双向点云 F-score 为主。

平均渲染质量如下：

| 指标 | 数值 |
|---|---:|
| 平均 PSNR | 26.88 dB |
| 平均 SSIM | 0.895 |
| 平均 LPIPS | 0.149 |

PSNR 达到 25dB 以上，说明整体新视角渲染质量良好；部分室内弱纹理区域和反光区域仍存在模糊与漂浮物。

### 3.2 定量结果

完整指标见 `metrics/summary.md`。各序列最终选择如下：

| Sequence | 最终方法 | GT->Recon@10cm | Recon->GT@10cm | F@10cm | GT->Recon@20cm | Recon->GT@20cm | F@20cm |
|---|---|---:|---:|---:|---:|---:|---:|
| Sequence_01 | GeoSVR + MVS | 0.989497 | 0.992867 | 0.991179 | 0.999450 | 0.999576 | 0.999513 |
| Sequence_02 | GeoSVR + MVS | 0.904157 | 0.906402 | 0.905278 | 0.983021 | 0.969358 | 0.976142 |
| Sequence_03 | PlanarGS + MVS | 0.997448 | 0.971742 | 0.984427 | 0.999943 | 0.984672 | 0.992249 |
| Sequence_04 | LingBot-Map | 0.855268 | 0.649923 | 0.738589 | 0.921425 | 0.792978 | 0.852390 |
| Sequence_05 | GeoSVR + MVS | 0.955354 | 0.906488 | 0.930279 | 0.990509 | 0.937382 | 0.963214 |

从结果可以看出，除 Sequence_04 外，最终结果均达到 20cm 阈值下较高的覆盖和精度；Sequence_01、Sequence_02、Sequence_03 和 Sequence_05 的 20cm F-score 均超过 0.96。Sequence_04 虽然仍明显低于其他序列，但 LingBot-Map 相比 COLMAP 系方法有显著提升。

### 3.3 方法对比分析

GeoSVR 和 PlanarGS 在多数场景下表现接近，但趋势有所不同：

| Sequence | GeoSVR + MVS F@20cm | PlanarGS + MVS F@20cm | 结论 |
|---|---:|---:|---|
| Sequence_01 | 0.999513 | 0.997812 | GeoSVR 略优 |
| Sequence_02 | 0.976142 | 0.892181 | GeoSVR 明显更稳 |
| Sequence_03 | 0.983360 | 0.992249 | PlanarGS 最优 |
| Sequence_04 | 0.461440 | 0.610186 | 二者均受位姿问题限制 |
| Sequence_05 | 0.963214 | 数据不完整 | GeoSVR 最优 |

最终策略是：除 Sequence_03 外优先采用 GeoSVR；Sequence_03 使用 PlanarGS；Sequence_04 使用 LingBot-Map。若只允许提交一个主方法，则可以选择 GeoSVR 作为主线，并在文档中说明 Sequence_04 的特殊处理。

### 3.4 失败案例

**Sequence_04 位姿困难**  
该序列中 COLMAP 系方法出现明显全局错位。GeoSVR 和 PlanarGS 即使加入深度或平面先验，也无法完全修复错误相机位姿带来的结构扭曲。表现为墙体重影、房间尺度不连续、局部点云悬浮。LingBot-Map 在该序列上显著提高了 10cm 和 20cm F-score，说明更稳定的位姿/地图估计比单纯调高 3DGS 训练轮数更关键。

**TSDF 融合不稳定**  
在多个序列中，TSDF 结果低于 MVS。主要原因是 TSDF 会把多个视角的深度误差累积到同一个体素场中，若深度边界或相机位姿有偏差，会造成表面膨胀、厚墙和边界模糊。

**弱纹理和透明/反光物体**  
白墙、玻璃、反光桌面等区域仍是主要误差来源。它们在图像中缺少稳定特征，深度估计和多视图一致性都会下降，导致局部点云稀疏或浮点。

## 4. Reproducibility

仓库结构如下：

```text
final_submission/
  README.md
  docs/
    technical_report.md
    reproduce_commands.md
    submission_checklist.md
    members.md
  metrics/
    summary.md
  scripts/
    evaluate_reconstruction.py
    extract_depth_point_cloud.py
    mvs_fuse_point_cloud.py
    tsdf_fuse_point_cloud.py
    make_pointcloud_overlay.py
  third_party/
    THIRD_PARTY.md
```

典型评估命令如下：

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

LingBot-Map 导出的 GLB 可直接评估：

```bash
python scripts/evaluate_reconstruction.py \
  --pred export.glb \
  --gt data/Sequence_04/gt/gt_pd.ply \
  --out-dir eval_outputs/Sequence_04_lingbot \
  --sample-align 300000 \
  --sample-eval 0 \
  --icp-iterations 300 \
  --pca-permutations
```

生成红蓝叠加可视化：

```bash
python scripts/make_pointcloud_overlay.py \
  --pred eval_outputs/Sequence_04_lingbot/pred_aligned.ply \
  --gt data/Sequence_04/gt/gt_pd.ply \
  --out eval_outputs/Sequence_04_lingbot/overlay_pred_red_gt_blue.ply
```

## 5. Conclusion

本项目通过调研 GeoSVR、PlanarGS 与 LingBot-Map，建立了一个面向室内 3DGS 重建的多方法评估与择优 pipeline。实验结果表明：在大多数序列中，GeoSVR 依靠更稳定的深度估计取得了较好的几何精度；PlanarGS 在平面结构强且位姿可靠的 Sequence_03 中表现最好；Sequence_04 的主要瓶颈是位姿问题，因此采用不强依赖 COLMAP 的 LingBot-Map 获得最佳效果。

自写评估脚本提供了统一、可复现的点云精度评估流程，能够自动完成相似变换对齐、双向最近邻距离计算、10cm/20cm F-score 统计和可视化输出。最终结果在多数序列上达到 20cm 与 10cm 精度要求，平均 PSNR 为 26.88dB。

不足之处在于：对极端位姿失败序列仍依赖替代建图方法；透明/反光和弱纹理区域仍存在局部缺失；TSDF 融合对位姿误差较敏感。后续可以继续引入更强的视觉定位、语义/平面联合约束，以及更鲁棒的深度不确定性建模。
