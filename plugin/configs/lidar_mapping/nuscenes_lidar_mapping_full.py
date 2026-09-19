import os
import math

_base_ = ['./nuscenes_lidar_mapping_5frame_mini.py']

# Set absolute paths before preprocessing and torchrun. Keep mini/full separate.
data_root = os.environ.get('NUSCENES_ROOT', './datasets/nuscenes_full')
ann_root = os.environ.get('NUSCENES_ANN_ROOT', data_root)
train_ann = os.path.join(ann_root, 'nuscenes_map_infos_train.pkl')
val_ann = os.path.join(ann_root, 'nuscenes_map_infos_val.pkl')
world_size = int(os.environ.get('WORLD_SIZE', '1'))
samples_per_gpu = 1
# Official old split usually has 27968 frames. Override if converter count differs.
train_frames = int(os.environ.get('NUSCENES_TRAIN_FRAMES', '27968'))
iters_per_epoch = math.ceil(train_frames / (world_size * samples_per_gpu))
train_epochs = int(os.environ.get('LIDAR_TRAIN_EPOCHS', '24'))

model = dict(backbone_cfg=dict(
    point_cloud_range=(-51.2, -51.2, -5., 51.2, 51.2, 3.),
    voxel_size=(0.1, 0.1, 0.2), max_voxels=(90000, 120000)))
eval_config = dict(data_root=data_root, ann_file=val_ann)
data = dict(
    samples_per_gpu=samples_per_gpu, workers_per_gpu=2,
    train=dict(data_root=data_root, ann_file=train_ann,
               matching_file=os.path.join(ann_root, 'nuscenes_map_infos_train_lidar_gt_tracks.pkl')),
    val=dict(data_root=data_root, ann_file=val_ann, eval_config=eval_config),
    test=dict(data_root=data_root, ann_file=val_ann, eval_config=eval_config))
# Converted frontend only, NOT the unmodified upstream checkpoint.
load_from = 'work_dirs/pretrained_ckpts/lidar-only-seg-maptracker.pth'
optimizer = dict(
    lr=1e-4, paramwise_cfg=dict(_delete_=True, custom_keys={
        'backbone.encoders': dict(lr_mult=0.1),
        'backbone.decoder': dict(lr_mult=0.1)}))
runner = dict(max_iters=train_epochs * iters_per_epoch)
lr_config = dict(warmup_iters=500)
checkpoint_config = dict(interval=iters_per_epoch, create_symlink=False)
evaluation = dict(interval=iters_per_epoch)
work_dir = 'work_dirs/lidar_mapping_full'
