_base_ = ['../bev_localization/nuscenes_lidar_localization_mini.py']

# Mapping only: no localization head, no prior-map query, no loc losses.
# Train individual frames, stream inference with pose-aligned vector queries.
model = dict(
    localization_cfg=None, localization_only=False, skip_vector_head=False,
    lidar_online_mapping=True, history_steps=1, test_time_history_steps=1,
    use_memory=False,
)
meta_keys = ('token', 'sample_idx', 'ego2global_translation',
             'ego2global_rotation', 'scene_name', 'observation_modality')
raster_pipeline = [dict(type='RasterizeMap', roi_size=(60, 30), coords_dim=2,
                        canvas_size=(200, 100), thickness=3, semantic_mask=True)]
train_pipeline = [
    dict(type='VectorizeMap', coords_dim=2, roi_size=(60, 30), sample_num=20,
         normalize=True, permute=True),
] + raster_pipeline + [
    dict(type='LoadNuScenesLidarForBEVFusion', shuffle=True),
    dict(type='FormatBundleMap', process_img=False),
    dict(type='Collect3D', keys=['points', 'vectors', 'semantic_mask'], meta_keys=meta_keys),
]
# GT raster is available for offline metrics; forward_test does not consume it.
test_pipeline = raster_pipeline + [
    dict(type='LoadNuScenesLidarForBEVFusion'),
    dict(type='FormatBundleMap', process_img=False),
    dict(type='Collect3D', keys=['points', 'semantic_mask'], meta_keys=meta_keys),
]
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    train=dict(localization_prior=None, pipeline=train_pipeline,
               multi_frame=False, matching=False),
    val=dict(localization_prior=None, pipeline=test_pipeline, eval_semantic=False),
    test=dict(localization_prior=None, pipeline=test_pipeline, eval_semantic=False),
)
load_from = None
runner = dict(max_iters=500)
checkpoint_config = dict(interval=100, create_symlink=False)
work_dir = 'work_dirs/lidar_mapping_mini'
