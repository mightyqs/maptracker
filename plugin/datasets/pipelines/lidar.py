"""Single-sweep nuScenes input for the BEVFusion-compatible LiDAR frontend."""
import numpy as np
from pyquaternion import Quaternion
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import LiDARPoints


@PIPELINES.register_module()
class LoadNuScenesLidarForBEVFusion:
    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def __call__(self, results):
        raw = np.fromfile(results['lidar_path'], dtype=np.float32).reshape(-1, 5)
        points = raw.copy()
        points[:, 4] = 0.0  # raw fifth column is ring index, NOT sweep time lag
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite points: {results['lidar_path']}")
        if self.shuffle:
            np.random.shuffle(points)
        results['points'] = LiDARPoints(points, points_dim=5,
                                       attribute_dims={'intensity': 3})
        # Preserve native point coordinates. Metadata describes the rotated map
        # frame used by the BEV adapter, consistent with the map extractor.
        ego_rotation = np.asarray(results['ego2global_rotation'])
        lidar_rotation = Quaternion(results['lidar2ego_rotation']).rotation_matrix
        lidar_translation = np.asarray(results['lidar2ego_translation'])
        ego_translation = np.asarray(results['raw_ego2global_translation'])
        lidar_global_translation = ego_translation + ego_rotation @ lidar_translation
        lidar_global_rotation = ego_rotation @ lidar_rotation
        yaw = np.arctan2(lidar_global_rotation[1, 0], lidar_global_rotation[0, 0])
        angle = yaw + np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        results['ego2global_translation'] = lidar_global_translation.tolist()
        results['ego2global_rotation'] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        results['observation_modality'] = 'lidar'
        return results
