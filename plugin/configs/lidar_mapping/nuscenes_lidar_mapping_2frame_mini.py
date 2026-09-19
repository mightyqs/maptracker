_base_ = ['./nuscenes_lidar_mapping_mini.py']

# Original MapTracker window sampler: one past frame from the previous two.
# Scene-start padding, GT association, query selection, detach and f/b losses
# retain upstream semantics. Each sweep has its own BEV; no BEV history fusion.
data = dict(train=dict(
    multi_frame=2, sampling_span=2, matching=True,
    matching_file='./datasets/nuscenes/nuscenes_map_infos_train_lidar_gt_tracks.pkl'))
work_dir = 'work_dirs/lidar_mapping_2frame_mini'
