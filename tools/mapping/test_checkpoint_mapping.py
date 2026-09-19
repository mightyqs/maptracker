import unittest
import torch
from convert_bevfusion_checkpoint import map_frontend


class CheckpointMappingTests(unittest.TestCase):
    def test_module_prefix_maps_only_frontend(self):
        tensor = torch.ones(2, 3)
        mapped = map_frontend({'module.encoders.lidar.backbone.weight': tensor}, {
            'backbone.encoders.lidar.backbone.weight': torch.zeros(2, 3),
            'backbone.adapter.weight': torch.zeros(2, 3)})
        self.assertEqual(list(mapped), ['backbone.encoders.lidar.backbone.weight'])
        torch.testing.assert_close(mapped[next(iter(mapped))], tensor)

    def test_missing_or_mismatched_kernel_fails(self):
        target = {'backbone.encoders.lidar.backbone.weight': torch.zeros(2, 3)}
        for source in ({}, {'encoders.lidar.backbone.weight': torch.ones(3, 2)},
                       {'encoders.lidar.backbone.weight': torch.full((2, 3), float('nan'))}):
            with self.assertRaises(ValueError):
                map_frontend(source, target)


if __name__ == '__main__':
    unittest.main()
