_base_ = ['./nuscenes_lidar_mapping_mini.py']

# Original BEVFusion voxel scope/resolution; still reads mini data by default.
model = dict(backbone_cfg=dict(
    point_cloud_range=(-51.2, -51.2, -5., 51.2, 51.2, 3.),
    voxel_size=(0.1, 0.1, 0.2), max_voxels=(90000, 120000)))
work_dir = 'work_dirs/lidar_mapping_bevfusion'
