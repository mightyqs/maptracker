_base_ = ['./nuscenes_raster_localization_real_crop.py']

# BEVFusion widths/depths and XYZ sparse layout, with a smaller XY grid for
# local pipeline checks. No image network, no pretrained weights, no history.
model = dict(
    freeze_bev=False, freeze_bev_iters=None,
    backbone_cfg=dict(
        _delete_=True, type='BEVFusionLidarBackbone',
        roi_size=(60., 30.), bev_h=50, bev_w=100,
        point_cloud_range=(-19.2, -32., -5., 19.2, 32., 3.),
        voxel_size=(0.1, 0.1, 0.2), max_num_points=10,
        max_voxels=(90000, 120000), freeze_encoder=False),
    localization_cfg=dict(detach_bev=False),
)

raster_pipeline = [
    dict(type='RasterizeMap', roi_size=(60, 30), coords_dim=2,
         canvas_size=(200, 100), thickness=3, semantic_mask=True),
    dict(type='RasterizeMap', roi_size=(60, 30), coords_dim=2,
         canvas_size=(200, 100), thickness=3, semantic_mask=True,
         geometry_key='localization_map_geoms', output_key='localization_map'),
]
meta_keys = ('token', 'sample_idx', 'ego2global_translation',
             'ego2global_rotation', 'scene_name', 'observation_modality',
             'localization_prior_global_pose', 'localization_observation_global_pose')
train_pipeline = [
    dict(type='VectorizeMap', coords_dim=2, roi_size=(60, 30), sample_num=20,
         normalize=True, permute=True),
] + raster_pipeline + [
    dict(type='LoadNuScenesLidarForBEVFusion', shuffle=True),
    dict(type='FormatBundleMap', process_img=False),
    dict(type='Collect3D', keys=['points', 'vectors', 'semantic_mask',
                               'localization_map', 'localization_target_pose'],
         meta_keys=meta_keys),
]
test_pipeline = raster_pipeline + [
    dict(type='LoadNuScenesLidarForBEVFusion'),
    dict(type='FormatBundleMap', process_img=False),
    dict(type='Collect3D', keys=['points', 'semantic_mask', 'localization_map',
                               'localization_target_pose'], meta_keys=meta_keys),
]
lidar_meta = dict(use_camera=False, use_lidar=True)
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    train=dict(pipeline=train_pipeline, meta=lidar_meta),
    val=dict(pipeline=test_pipeline, meta=lidar_meta),
    test=dict(pipeline=test_pipeline, meta=lidar_meta),
)
load_from = None
resume_from = None
optimizer = dict(lr=1e-4)
runner = dict(max_iters=500)
checkpoint_config = dict(interval=100, create_symlink=False)
work_dir = 'work_dirs/lidar_localization_mini'
