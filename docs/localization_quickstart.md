# Raster-map localization quick validation

This branch adds a small localization head to MapTracker. It shares the fused
BEV tensor with the segmentation and vector heads and matches it against an
encoded nuScenes vector-map raster.

The first milestone deliberately uses synthetic prior errors at the feature
level. It validates the coordinate convention, joint SE(2) probability volume,
losses, covariance, and MapTracker integration. It is not a global
relocalization benchmark and does not yet use an accumulated LiDAR map.

## Components

- `LocalizationNeck`: projects `[B, 256, 50, 100]` MapTracker BEV features to
  compact normalized descriptors.
- `RasterMapEncoder`: encodes the existing three-channel nuScenes semantic map.
- `SemanticDecoder`: reconstructs the map raster from map descriptors and
  predicts the same semantics from observation descriptors during training.
- `SE2TemplateMatcher`: evaluates a joint local `x/y/yaw` hypothesis grid.
- `RasterMapLocalizationHead`: samples a known prior error and supervises the
  pose probability volume.

Pose convention is forward `x`, left `y`, and counter-clockwise yaw. The head
returns MAP/mean pose, a `3x3` covariance, entropy, peak ratio, and confidence.

## Pure PyTorch smoke test

The smoke test does not import MMCV or MMDetection:

```bash
python tools/localization/smoke_test.py --device cuda
```

It creates an exact synthetic prior error, runs the joint matcher, verifies the
correct hypothesis, and checks gradient propagation.

## nuScenes training

Prepare nuScenes and the official MapTracker annotations as described in
`docs/data_preparation.md`. Download the official stage-3 nuScenes checkpoint
to:

```text
work_dirs/pretrained_ckpts/
  maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth
```

Then run the single-GPU quick config:

```bash
bash tools/dist_train.sh \
  plugin/configs/bev_localization/nuscenes_raster_localization_quick.py 1
```

This config freezes BEVFormer, disables multi-frame/vector/segmentation
training, keeps the official `480x800` image size, and trains the localization
neck, map encoder, matcher, and two auxiliary semantic decoders. The visual
semantic loss reaches the localization neck but not the frozen BEVFormer in
this quick-validation stage. Set both `freeze_bev=False` and `detach_bev=False`
for later joint fine-tuning.
Useful log fields are:

- `loc_nll` and `loc_reg`: optimized localization losses.
- `map_recon_focal` and `map_recon_dice`: semantic-map reconstruction losses.
- `bev_sem_focal` and `bev_sem_dice`: visual BEV semantic losses.
- `loc_map_recon_miou` and `loc_bev_sem_miou`: training-time semantic mIoU.
- `loc_map_recon_iou_c*` and `loc_bev_sem_iou_c*`: per-channel IoU, where
  `c0/c1/c2` are pedestrian crossing, divider, and boundary respectively.
- `loc_exact_acc`: exact hypothesis classification accuracy.
- `loc_err_x_m`, `loc_err_y_m`, `loc_err_yaw_deg`: MAP hypothesis errors.

## Next milestone

The real-map crop path is now available in
[`localization_real_crop.md`](localization_real_crop.md), including training and
paired evaluation commands. After that is stable, replace the semantic raster input with
multi-channel rasters accumulated from a disjoint nuScenes LiDAR map session.

## mini-val 固定扰动与错配对照

`tools/localization/evaluate_fixed.py` 提供独立的定位评估入口，不依赖原有的
地图分割/向量评估器。仅支持当前单帧、无历史记忆、FP32 的定位配置。

在 WSL 的 `maptracker` 环境运行：

```bash
conda activate maptracker
cd /mnt/e/source_code/maptracker
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python tools/localization/evaluate_fixed.py \
  plugin/configs/bev_localization/nuscenes_raster_localization_quick.py \
  work_dirs/localization_mini_500iter/iter_100.pth \
  work_dirs/localization_mini_500iter/iter_300.pth \
  work_dirs/localization_mini_500iter/iter_500.pth \
  --out-dir work_dirs/localization_mini_val_fixed_run2 \
  --repeats 5 --seed 20260916
```

输出目录必须为空，避免覆盖已有实验。此命令不会训练或更新权重。

- 全部 81 帧，每帧从当前 245 个候选中固定抽取 5 个不同扰动，共 405 组。
  使用 seed、sample token 与候选编号的稳定哈希，多个 checkpoint 共用同一清单。
- `correct`：本帧六相机图像，与本帧地图匹配。
- `mismatched`：另一个验证场景的六相机图像，与本帧地图匹配。先用供体自身
  相机几何编码，再替换观测描述子；不混用供体图像和接收帧相机外参。
  供体按目标帧和扰动序号确定，允许重复使用，并非一对一排列。
- `identity`：预测零修正，即不修正初始位姿。地图、目标扰动在各条件间一致。
- 显式检查当前训练/验证标注的场景无交集；此检查不证明上游预训练权重从未
  见过验证场景。

`manifest.json` 保存帧、供体、候选编号与目标位姿；每个 checkpoint 有逐组
`*_records.json` 和 `*_summary.json`，总表为 `summary.json`。
统计包括 x/y MAE、平移 L2 误差均值/中位数/P95、yaw 误差、精确命中率、
成功率（默认平移 ≤1 m 且 yaw ≤1°）、概率均值位姿误差，以及各场景结果。
`paired.same_map_prediction_rate` 表示更换图像后 MAP 预测完全不变的比例。
MAP 与均值误差都比较预测修正和目标修正，不是绝对全局轨迹误差。

语义 IoU 是正确配对时在整个验证集累计 intersection/union 后计算的指标；
训练日志则是 batch IoU 的窗口平均，两者统计口径不同。定位置信度仍是熵派生
量，不是经过校准的成功概率。405 组包含重复帧和同场景相关帧，不能视为
405 个相互独立场景。

此评估仍使用**地图特征级合成扰动**。若错配后依旧高准确率，不能将正确配对
成绩解释为已经学会图像—地图定位；应优先检查合成变换边界、插值和位置特征
捷径，再验证带噪全局位姿下的真实地图裁剪。

协议测试：

```bash
python -m unittest discover -s tools/localization -p test_evaluate_fixed.py -v
```
