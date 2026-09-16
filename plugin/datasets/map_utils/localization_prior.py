"""SE(2) conventions for querying a real nuScenes map at an uncertain pose."""
import hashlib
import math

import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion


def se2_matrix(pose):
    x, y, yaw = np.asarray(pose, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def se2_pose(matrix):
    return np.array([matrix[0, 2], matrix[1, 2],
                     math.atan2(matrix[1, 0], matrix[0, 0])])


def lidar_global_pose(sample):
    ego = Quaternion(sample['e2g_rotation'])
    lidar = Quaternion(sample['lidar2ego_rotation'])
    translation = ego.rotate(sample['lidar2ego_translation']) + np.asarray(sample['e2g_translation'])
    return translation, (ego * lidar).elements


def prior_map_query(translation, rotation, target_pose):
    """Return extractor query and auditable observation/prior global SE(2).

    The existing extractor rotates LiDAR-local geometries by -90 degrees.
    Therefore its OUTPUT raster frame has global yaw lidar_yaw + pi/2.
    C=target_pose maps observation coordinates into prior-map coordinates,
    exactly the output-to-input convention of build_sampling_grid.
    P = G @ inverse(C); a predicted C restores global pose as P @ C.
    """
    target_pose = np.asarray(target_pose, dtype=np.float64)
    if target_pose.shape != (3,) or not np.isfinite(target_pose).all():
        raise ValueError('target_pose must be a finite [x, y, yaw] vector')
    global_pose = np.array([translation[0], translation[1],
                           quaternion_yaw(Quaternion(rotation)) + math.pi / 2])
    prior = se2_matrix(global_pose) @ np.linalg.inv(se2_matrix(target_pose))
    prior_pose = se2_pose(prior)
    query_translation = [float(prior_pose[0]), float(prior_pose[1]), float(translation[2])]
    query_rotation = Quaternion(axis=[0, 0, 1], angle=prior_pose[2] - math.pi / 2).elements.tolist()
    # Preserve the exact original query at identity (including roll/pitch).
    if np.array_equal(target_pose, np.zeros(3)):
        query_translation, query_rotation = list(translation), list(rotation)
    return dict(translation=query_translation, rotation=query_rotation,
                observation_global_pose=se2_pose(se2_matrix(global_pose)).tolist(),
                prior_global_pose=prior_pose.tolist())


def sample_correction(config, token, test_mode):
    """Train: fresh RNG draws; validation: token-stable draws without RNG effects."""
    rng = np.random
    if test_mode:
        key = f'{config.get("seed", 20260916)}:{token}'.encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')
        rng = np.random.RandomState(seed)
    mode = config.get('sampling', 'grid')
    max_translation = float(config.get('max_translation', 3.6))
    max_yaw = float(config.get('max_yaw_deg', 4.0))
    if max_translation < 0 or max_yaw < 0:
        raise ValueError('Perturbation ranges must be non-negative')
    if mode == 'continuous':
        return np.array([rng.uniform(-max_translation, max_translation),
                         rng.uniform(-max_translation, max_translation),
                         math.radians(rng.uniform(-max_yaw, max_yaw))], dtype=np.float32)
    if mode != 'grid':
        raise ValueError(f'Unknown prior sampling mode: {mode}')
    xy_step = float(config.get('translation_step', 1.2))
    yaw_step = float(config.get('yaw_step_deg', 2.0))
    if xy_step <= 0 or yaw_step <= 0:
        raise ValueError('Grid steps must be positive')
    xy_count = int(math.floor(max_translation / xy_step + 1e-6))
    yaw_count = int(math.floor(max_yaw / yaw_step + 1e-6))
    return np.array([rng.randint(-xy_count, xy_count + 1) * xy_step,
                     rng.randint(-xy_count, xy_count + 1) * xy_step,
                     math.radians(rng.randint(-yaw_count, yaw_count + 1) * yaw_step)],
                    dtype=np.float32)
