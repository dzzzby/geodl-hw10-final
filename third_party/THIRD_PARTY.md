# Third-Party Code

本项目没有直接修改第三方仓库源码；最终提交目录只保存自写评估脚本、指标表和技术文档。第三方方法通过独立仓库运行，输出点云后再进入统一评估流程。

## GeoSVR

- Repository: https://github.com/Fictionarry/GeoSVR
- 用途：主要 3DGS 重建方法。利用更稳定的深度估计/几何约束改善室内弱纹理区域。
- 本项目选择：Sequence_01、Sequence_02、Sequence_05 采用 `geosvr+mvs`；Sequence_03/04 作为对比方法。

建议放置：

```bash
git clone https://github.com/Fictionarry/GeoSVR third_party/GeoSVR
```

## PlanarGS

- Repository: https://github.com/SJTU-ViSYS-team/PlanarGS
- 用途：室内平面先验 3DGS，对墙面、地面等平面结构有帮助。
- 本项目选择：Sequence_03 采用 `planargs+mvs`，其他序列作为对比。

建议放置：

```bash
git clone https://github.com/SJTU-ViSYS-team/PlanarGS third_party/PlanarGS
```

## LingBot-Map

- Repository: https://github.com/robbyant/lingbot-map
- 用途：用于传统 COLMAP 位姿不稳定的序列，尤其是 Sequence_04。
- 本项目选择：Sequence_04 采用 `lingbot-map`。

建议放置：

```bash
git clone https://github.com/robbyant/lingbot-map third_party/lingbot-map
```

## Papers

本项目参考论文位于原工作目录：

```text
D:\GCL\26summerGDC\papers
```

包括：

- `geosvr.pdf`
- `PlanarGS.pdf`
- `lingbot-map.pdf`
- `LongSplat.pdf`
- `survey-how.pdf`

这些 PDF 体积较大，未复制进最终提交目录；报告中只保留方法说明和代码来源。
