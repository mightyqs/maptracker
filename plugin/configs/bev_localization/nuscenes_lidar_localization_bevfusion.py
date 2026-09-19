_base_ = ['./nuscenes_lidar_localization_mini.py']

# Native upstream lidar-centerpoint-bev128 voxel range/resolution and encoder.
# Still single-sweep, with MapTracker BEV adapter and localization heads.
# This changes frontend resolution, NOT the dataset split (still mini by default).
model = dict(backbone_cfg=dict(
    point_cloud_range=(-51.2, -51.2, -5., 51.2, 51.2, 3.),
    voxel_size=(0.1, 0.1, 0.2), max_voxels=(90000, 120000)))
work_dir = 'work_dirs/lidar_localization_bevfusion'
