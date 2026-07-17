_base_ = [
    '../../maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage1_bev_pretrain.py'
]

# This config validates the localization branch with synthetic SE(2) prior
# errors on nuScenes vector-map rasters. It is intentionally small enough for
# a single low-memory GPU and is not a final localization benchmark.

img_size = (480, 800)
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False,
)

model = dict(
    skip_vector_head=True,
    freeze_bev=True,
    use_memory=False,
    history_steps=0,
    test_time_history_steps=0,
    localization_only=True,
    localization_cfg=dict(
        type='RasterMapLocalizationHead',
        bev_in_channels=256,
        map_in_channels=3,
        hidden_channels=64,
        descriptor_dim=32,
        roi_size=(60.0, 30.0),
        max_translation=3.6,
        translation_step=1.2,
        max_yaw_deg=4.0,
        yaw_step_deg=2.0,
        candidate_chunk_size=32,
        regression_loss_weight=0.25,
        synthetic_train_perturbation=True,
        synthetic_test_perturbation=True,
        detach_bev=True,
        loss_weight=1.0,
        map_reconstruction_loss_weight=1.0,
        bev_semantic_loss_weight=1.0,
        semantic_focal_loss_weight=1.0,
        semantic_dice_loss_weight=1.0,
        semantic_decoder_hidden_channels=64,
    ),
)

train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=2,
        roi_size=(60, 30),
        sample_num=20,
        normalize=True,
        permute=True,
    ),
    dict(
        type='RasterizeMap',
        roi_size=(60, 30),
        coords_dim=2,
        canvas_size=(200, 100),
        thickness=3,
        semantic_mask=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(
        type='Collect3D',
        keys=['img', 'vectors', 'semantic_mask'],
        meta_keys=(
            'token',
            'ego2img',
            'sample_idx',
            'ego2global_translation',
            'ego2global_rotation',
            'img_shape',
            'scene_name',
        ),
    ),
]

test_pipeline = [
    dict(
        type='RasterizeMap',
        roi_size=(60, 30),
        coords_dim=2,
        canvas_size=(200, 100),
        thickness=3,
        semantic_mask=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(
        type='Collect3D',
        keys=['img', 'semantic_mask'],
        meta_keys=(
            'token',
            'ego2img',
            'sample_idx',
            'ego2global_translation',
            'ego2global_rotation',
            'img_shape',
            'scene_name',
        ),
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        multi_frame=False,
        matching=False,
        seq_split_num=-2,
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-2,
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    min_lr_ratio=0.05,
)

runner = dict(type='MyRunnerWrapper', max_iters=2000)
checkpoint_config = dict(interval=500)
evaluation = dict(interval=2000)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

find_unused_parameters = True
SyncBN = False

# Download this checkpoint using docs/data_preparation.md before training.
load_from = (
    'work_dirs/pretrained_ckpts/'
    'maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth'
)
