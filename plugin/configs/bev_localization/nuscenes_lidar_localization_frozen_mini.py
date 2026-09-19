_base_ = ['./nuscenes_lidar_localization_mini.py']

# Initialize with the completed LiDAR mapping checkpoint using load_from.
model = dict(
    freeze_bev=True, freeze_mapping_for_localization=True,
    localization_only=True, skip_vector_head=True,
    history_steps=0, test_time_history_steps=0, use_memory=False,
    localization_cfg=dict(detach_bev=True))
work_dir = 'work_dirs/lidar_localization_frozen_mini'
