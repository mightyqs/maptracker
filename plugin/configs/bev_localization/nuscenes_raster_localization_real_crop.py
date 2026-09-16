_base_ = ['./nuscenes_raster_localization_quick.py']

# Query nuScenes global map at a noisy prior BEFORE rasterization/encoding.
# Keep the old feature-level config intact for controlled comparisons.
model = dict(localization_cfg=dict(
    synthetic_train_perturbation=False,
    synthetic_test_perturbation=False,
    require_prior_map=True,
))
localization_prior = dict(
    max_translation=3.6, translation_step=1.2,
    max_yaw_deg=4.0, yaw_step_deg=2.0,
    sampling='grid', seed=20260916,
)
raster_pipeline = [
    dict(type='RasterizeMap', roi_size=(60, 30), coords_dim=2,
         canvas_size=(200, 100), thickness=3, semantic_mask=True),
    dict(type='RasterizeMap', roi_size=(60, 30), coords_dim=2,
         canvas_size=(200, 100), thickness=3, semantic_mask=True,
         geometry_key='localization_map_geoms', output_key='localization_map'),
]
image_pipeline = [
    dict(type='ResizeMultiViewImages', size=(480, 800), change_intrinsics=True),
    dict(type='Normalize3D', mean=[103.530, 116.280, 123.675],
         std=[1.0, 1.0, 1.0], to_rgb=False),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
]
meta_keys = ('token', 'ego2img', 'sample_idx', 'ego2global_translation',
             'ego2global_rotation', 'img_shape', 'scene_name',
             'localization_prior_global_pose', 'localization_observation_global_pose')
train_pipeline = [
    dict(type='VectorizeMap', coords_dim=2, roi_size=(60, 30), sample_num=20,
         normalize=True, permute=True),
] + raster_pipeline + [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
] + image_pipeline + [
    dict(type='Collect3D', keys=['img', 'vectors', 'semantic_mask',
                               'localization_map', 'localization_target_pose'],
         meta_keys=meta_keys),
]
test_pipeline = raster_pipeline + [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
] + image_pipeline + [
    dict(type='Collect3D', keys=['img', 'semantic_mask', 'localization_map',
                               'localization_target_pose'], meta_keys=meta_keys),
]
data = dict(
    # Use batch 1 via CLI when the GPU is shared with other applications.
    samples_per_gpu=4, workers_per_gpu=2,
    train=dict(localization_prior=localization_prior, pipeline=train_pipeline),
    val=dict(localization_prior=localization_prior, pipeline=test_pipeline),
    test=dict(localization_prior=localization_prior, pipeline=test_pipeline),
)
load_from = 'work_dirs/localization_mini_500iter/iter_500.pth'
runner = dict(max_iters=500)
lr_config = dict(warmup_iters=50)
checkpoint_config = dict(interval=100, create_symlink=False)
work_dir = 'work_dirs/localization_mini_real_crop'
