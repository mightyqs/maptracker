_base_ = ['./nuscenes_lidar_mapping_2frame_mini.py']

# Same five-frame / span-ten sampling as the upstream nuScenes configuration.
model = dict(history_steps=4)
data = dict(train=dict(multi_frame=5, sampling_span=10))
work_dir = 'work_dirs/lidar_mapping_5frame_mini'
