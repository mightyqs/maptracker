"""Geometry, external supervision, and real global-query regression tests."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugin.datasets.map_utils.localization_prior import (
    prior_map_query, sample_correction, se2_matrix,
)
from plugin.datasets.nusc_dataset import NuscDataset
from plugin.models.localization.core import SE2TemplateMatcher, SyntheticLocalizationCore
from plugin.models.localization.head import RasterMapLocalizationHead


class RealCropGeometryTest(unittest.TestCase):
    def test_global_composition_and_point_coordinates(self):
        for yaw in (0., 0.7, -2.5):
            rotation = Quaternion(axis=[0, 0, 1], angle=yaw - math.pi / 2).elements
            for correction in ([1.2, -2.4, 0.07], [-3.6, 1.2, -0.04]):
                query = prior_map_query([100., 200., 3.], rotation, correction)
                prior = se2_matrix(query['prior_global_pose'])
                observation = se2_matrix(query['observation_global_pose'])
                np.testing.assert_allclose(prior @ se2_matrix(correction), observation, atol=1e-12)
                point = np.array([8., -3., 1.])
                # Independent world-to-prior point projection must agree with C.
                np.testing.assert_allclose(np.linalg.solve(prior, observation @ point),
                                           se2_matrix(correction) @ point, atol=1e-12)

    def test_zero_query_preserves_original_rotation_and_height(self):
        rotation = Quaternion(axis=[1, 2, 3], angle=0.5).elements.tolist()
        query = prior_map_query([123., 456., 2.5], rotation, [0., 0., 0.])
        self.assertEqual(query['translation'], [123., 456., 2.5])
        self.assertEqual(query['rotation'], rotation)

    def test_forward_translation_sign_and_extractor_ninety_degrees(self):
        q = Quaternion(axis=[0, 0, 1], angle=-math.pi / 2).elements
        query = prior_map_query([0, 0, 0], q, [3.6, 0, 0])
        np.testing.assert_allclose(query['translation'], [-3.6, 0, 0], atol=1e-12)

    def test_real_query_can_include_geometry_outside_gt_crop(self):
        from nuscenes.eval.common.utils import quaternion_yaw

        class GlobalMap:
            def get_map_geom(self, location, translation, rotation):
                # A landmark outside the GT [-30,30] window, but inside the prior.
                landmark = LineString([(-31, -2), (-31, 2)])
                yaw = quaternion_yaw(Quaternion(rotation)) + math.pi / 2
                local = affinity.translate(landmark, -translation[0], -translation[1])
                local = affinity.rotate(local, -yaw, origin=(0, 0), use_radians=True)
                clipped = local.intersection(box(-30, -15, 30, 15))
                return {'divider': [] if clipped.is_empty else [clipped]}

        dataset = NuscDataset.__new__(NuscDataset)
        dataset.samples = [dict(location='fake', e2g_rotation=[1, 0, 0, 0],
                                e2g_translation=[0, 0, 0], lidar2ego_translation=[0, 0, 0],
                                lidar2ego_rotation=Quaternion(axis=[0, 0, 1],
                                                              angle=-math.pi / 2).elements)]
        dataset.map_extractor = GlobalMap()
        dataset.cat2id = {'divider': 0}
        gt = dataset.get_localization_prior(0, [0, 0, 0])
        noisy = dataset.get_localization_prior(0, [3.6, 0, 0])
        self.assertEqual(gt['localization_map_geoms'][0], [])
        self.assertEqual(len(noisy['localization_map_geoms'][0]), 1)
        self.assertAlmostEqual(noisy['localization_map_geoms'][0][0].bounds[0], -27.4)

    def test_validation_sampling_is_stable_and_training_varies(self):
        for mode in ('grid', 'continuous'):
            cfg = dict(sampling=mode, seed=123)
            np.testing.assert_array_equal(sample_correction(cfg, 'token', True),
                                          sample_correction(cfg, 'token', True))
            draws = [tuple(sample_correction(cfg, 'token', False)) for _ in range(10)]
            self.assertGreater(len(set(draws)), 1)


class ExternalTargetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.core = SyntheticLocalizationCore(SE2TemplateMatcher(
            roi_size=(12, 8), max_translation=1, translation_step=1,
            max_yaw_deg=2, yaw_step_deg=2))
        self.observation = torch.randn(2, 4, 8, 12)
        self.map = torch.randn_like(self.observation)

    def test_external_labels_match_legacy_synthetic_loss(self):
        indices = torch.tensor([3, 18])
        losses, output = self.core(self.observation, self.map, True, indices)
        prior, _, target = self.core.synthesize_prior_error(self.map, indices)
        external_losses, external_output = self.core(
            self.observation, prior, target_pose=target)
        torch.testing.assert_close(output['logits'], external_output['logits'])
        for name in losses:
            torch.testing.assert_close(losses[name], external_losses[name])

    def test_continuous_target_backprop_and_double_perturbation_guard(self):
        target = torch.tensor([[.25, -.3, .01], [-.7, .4, -.02]])
        observation = self.observation.requires_grad_()
        losses, _ = self.core(observation, self.map, target_pose=target)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(observation.grad).all())
        self.assertGreater(observation.grad.abs().sum(), 0)
        with self.assertRaises(ValueError):
            self.core(observation, self.map, synthesize_error=True, target_pose=target)

    def test_separate_semantic_supervision(self):
        head = RasterMapLocalizationHead(
            bev_in_channels=8, hidden_channels=8, descriptor_dim=4,
            roi_size=(12, 8), max_translation=1, translation_step=1,
            max_yaw_deg=2, yaw_step_deg=2, synthetic_train_perturbation=False,
            require_prior_map=True, map_reconstruction_loss_weight=1,
            bev_semantic_loss_weight=1)
        bev = torch.randn(1, 8, 16, 24)
        prior = torch.zeros(1, 3, 32, 48)
        target = torch.zeros(1, 3)
        losses_a, output_a = head(bev, prior, target_pose=target,
                                 bev_semantic_target=torch.zeros_like(prior))
        losses_b, output_b = head(bev, prior, target_pose=target,
                                 bev_semantic_target=torch.ones_like(prior))
        # Changing GT semantics must not change matching logits or map supervision.
        torch.testing.assert_close(output_a['logits'], output_b['logits'])
        torch.testing.assert_close(losses_a['map_recon_focal'], losses_b['map_recon_focal'])
        self.assertNotEqual(float(losses_a['bev_sem_focal']), float(losses_b['bev_sem_focal']))
        with self.assertRaises(ValueError):
            head(bev, prior, target_pose=target)


if __name__ == '__main__':
    unittest.main()
