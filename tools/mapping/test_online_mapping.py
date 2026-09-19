import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from mmcv import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugin.models.mapers.MapTracker import MapTracker
from plugin.models.heads.MapDetectorHead import MapDetectorHead
from verify_lidar import update_global_map


class IdentityQuery(nn.Module):
    def forward(self, features, pose):
        return features


class OnlineMappingTests(unittest.TestCase):
    def test_gt_track_matching_handles_reorder_birth_disappearance_and_classes(self):
        model = MapTracker.__new__(MapTracker)
        nn.Module.__init__(model)
        previous = {0: {0: 7, 1: 8}, 1: {0: 7}}
        current = {0: {0: 8, 1: 9}, 1: {0: 7}}
        prev_gts = [dict(gt2local=[{0: (0, 0), 1: (0, 1), 2: (1, 0)}],
                         local2gt=[{(0, 0): 0, (0, 1): 1, (1, 0): 2}])]
        curr_gts = [dict(gt2local=[{0: (0, 0), 1: (0, 1), 2: (1, 0)}])]
        cur2prev, prev2cur = model.get_two_frame_matching(
            [previous], [current], prev_gts, curr_gts)
        torch.testing.assert_close(cur2prev[0], torch.tensor([1., -1., 2.]))
        torch.testing.assert_close(prev2cur[0], torch.tensor([-1., 0., 2.]))

    def test_track_ids_and_newborn_ids_share_mask_device(self):
        head = MapDetectorHead.__new__(MapDetectorHead)
        nn.Module.__init__(head)
        head.loss_cls = SimpleNamespace(use_sigmoid=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        def predictions(n):
            return dict(lines=[torch.zeros(n, 40, device=device)],
                        scores=[torch.ones(n, 3, device=device)],
                        hs_embeds=torch.zeros(1, n, 512, device=device))
        first = head.prepare_temporal_propagation(predictions(100), 'scene', 0)
        second = head.prepare_temporal_propagation(predictions(200), 'scene', 1)
        np.testing.assert_array_equal(first['global_ids'], np.arange(100))
        np.testing.assert_array_equal(second['global_ids'], np.arange(200))
        self.assertEqual(second['num_instance'], 200)

    def test_global_map_latest_uses_pose_and_scene_local_ids(self):
        storage = {}
        result = dict(pos_results=dict(
            scene_name='a', local_idx=0, vectors=np.array([[.5, .5, .6, .5]]),
            global_ids=[0], labels=[1], scores=[.8]),
            meta=dict(ego2global_translation=[10., 20., 0.],
                      ego2global_rotation=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
        update_global_map(storage, result, (60, 30))
        np.testing.assert_allclose(storage['a']['0']['vector'], [[10., 20.], [10., 26.]])
        result['pos_results']['local_idx'] = 1
        update_global_map(storage, result, (60, 30))
        self.assertEqual(storage['a']['0']['observations'], 2)
        self.assertEqual(storage['a']['0']['first_frame'], 0)
        self.assertEqual(storage['a']['0']['last_frame'], 1)
        result['pos_results']['scene_name'] = 'b'
        update_global_map(storage, result, (60, 30))
        self.assertEqual(len(storage), 2)

    def test_mapping_config_has_no_localization_or_camera_inputs(self):
        cfg = Config.fromfile('plugin/configs/lidar_mapping/nuscenes_lidar_mapping_mini.py')
        self.assertIsNone(cfg.model.localization_cfg)
        self.assertFalse(cfg.model.skip_vector_head)
        for split in ('train', 'val', 'test'):
            data = cfg.data[split]
            self.assertIsNone(data.localization_prior)
            collect = data.pipeline[-1]
            self.assertNotIn('img', collect['keys'])
            self.assertFalse(any(k.startswith('localization') for k in collect['keys']))

    def test_relative_pose_and_propagated_vector_geometry(self):
        # Previous frame at (10,20), current at (11,22) rotated +90 degrees.
        model = MapTracker.__new__(MapTracker)
        nn.Module.__init__(model)
        model.roi_size = (60., 30.)
        model.register_buffer('plane', torch.tensor([[[0., 0., 0., 1.]]], dtype=torch.float64))
        model.query_propagate = IdentityQuery()
        model.training = False
        previous = [dict(ego2global_translation=[10., 20., 0.],
                         ego2global_rotation=np.eye(3).tolist())]
        current = [dict(ego2global_translation=[11., 22., 0.],
                        ego2global_rotation=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])]
        c2p, p2c, _ = model.process_history_info(current, [previous])
        tracks = [dict(track_query_boxes=torch.tensor([[.5, .5, .6, .5]]),
                       track_query_hs_embeds=torch.zeros(1, 4))]
        model.temporal_propagate(torch.zeros(1, 4, 1, 1), current, c2p, p2c,
                                 False, tracks, get_trans_loss=False)
        # Previous (0,0),(6,0) -> current (-2,1),(-2,-5), then normalize.
        expected = torch.tensor([[28/60, 16/30, 28/60, 10/30]])
        torch.testing.assert_close(tracks[0]['trans_track_query_boxes'], expected)


if __name__ == '__main__':
    unittest.main()
