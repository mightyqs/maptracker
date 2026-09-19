_base_ = ['./nuscenes_lidar_localization_mini.py']

# Same single-frame frontend; restore original semantic/vector training heads.
# Temporal query propagation remains disabled for this LiDAR baseline.
model = dict(skip_vector_head=False, localization_only=False)
work_dir = 'work_dirs/lidar_joint_mini'
