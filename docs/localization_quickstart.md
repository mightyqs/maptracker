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
training, keeps the official `480x800` image size, and trains only the
localization branch.
Useful log fields are:

- `loc_nll` and `loc_reg`: optimized localization losses.
- `loc_exact_acc`: exact hypothesis classification accuracy.
- `loc_err_x_m`, `loc_err_y_m`, `loc_err_yaw_deg`: MAP hypothesis errors.

## Next milestone

Replace feature-level synthetic perturbation with a raster crop rendered at a
noisy global pose. After that is stable, replace the semantic raster input with
multi-channel rasters accumulated from a disjoint nuScenes LiDAR map session.
