"""Geometry/data contracts; optional upstream numerical parity on CUDA."""
import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugin.models.backbones.bevfusion_lidar import (
    BEVFusionLidarBackbone, build_xyz_sparse_encoder, map_grid_in_lidar)
from plugin.datasets.pipelines.lidar import LoadNuScenesLidarForBEVFusion


class LidarGeometryTest(unittest.TestCase):
    def test_native_xy_axes_to_map_endpoint_grid(self):
        # Non-square native scope prevents accidentally passing an XY swap.
        bounds = (-19.2, -32., -5., 19.2, 32., 3.)
        x = torch.linspace(-18.4, 18.4, 24)
        y = torch.linspace(-31.2, 31.2, 40)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        native = torch.stack((xx, yy)).unsqueeze(0)
        grid = map_grid_in_lidar((60., 30.), 50, 100, bounds)
        out = F.grid_sample(native, grid, align_corners=False)
        map_y = torch.linspace(15., -15., 50)[:, None].expand(50, 100)
        map_x = torch.linspace(-30., 30., 100)[None, :].expand(50, 100)
        torch.testing.assert_close(out[0, 0], -map_y, atol=5e-6, rtol=1e-6)
        torch.testing.assert_close(out[0, 1], map_x, atol=5e-6, rtol=1e-6)

    def test_intensity_preserved_ring_not_used_as_time(self):
        raw = np.array([[1., 2., 3., 42., 17.], [-2., 4., 1., 99., 5.]], np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'scan.bin')
            raw.tofile(path)
            output = LoadNuScenesLidarForBEVFusion()(dict(
                lidar_path=path, ego2global_rotation=np.eye(3).tolist(),
                raw_ego2global_translation=[10., 20., 0.],
                lidar2ego_rotation=[1., 0., 0., 0.],
                lidar2ego_translation=[1., 0., 2.]))
        np.testing.assert_array_equal(output['points'].tensor[:, :4], raw[:, :4])
        np.testing.assert_array_equal(output['points'].tensor[:, 4], [0., 0.])
        np.testing.assert_allclose(output['ego2global_translation'], [11., 20., 2.])
        np.testing.assert_allclose(np.asarray(output['ego2global_rotation'])[:2, :2],
                                   [[0., -1.], [1., 0.]], atol=1e-7)

    def test_voxel_xyz_order_and_mean(self):
        # Test the real MMCV voxelizer, without allocating a full backbone.
        from mmcv.ops import Voxelization
        from torch import nn
        frontend = BEVFusionLidarBackbone.__new__(BEVFusionLidarBackbone)
        nn.Module.__init__(frontend)
        frontend.encoders = nn.ModuleDict({'lidar': nn.ModuleDict({
            'voxelize': Voxelization([1., 1., 1.], [0., 0., 0., 8., 8., 8.], 10, 20)})})
        cloud = torch.tensor([[1.1, 2.1, 3.1, 10., 0.],
                              [1.3, 2.3, 3.3, 30., 0.]])
        features, coords = frontend.voxelize([cloud, cloud])
        torch.testing.assert_close(features, cloud.mean(0).repeat(2, 1))
        self.assertEqual(coords.tolist(), [[0, 1, 2, 3], [1, 1, 2, 3]])
        with self.assertRaisesRegex(ValueError, 'No valid LiDAR'):
            frontend.voxelize([cloud, cloud[:0]])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA sparse parity test')
    def test_sparse_encoder_against_upstream_source(self):
        upstream = Path(os.environ.get('BEVFUSION_SOURCE', '../bevfusion'))
        path = upstream / 'mmdet3d/models/backbones/sparse_encoder.py'
        if not path.exists():
            self.skipTest('Set BEVFUSION_SOURCE to the upstream checkout')
        # Evaluate upstream class in isolation: remove registry decoration and
        # substitute ONLY its sparse backend import with the installed backend.
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
            if isinstance(node, ast.ImportFrom) and node.module == 'mmdet3d.ops':
                if any(a.name == 'spconv' for a in node.names):
                    node.module = 'mmdet3d.models.middle_encoders'
                    node.names = [ast.alias(name='sparse_encoder', asname='spconv')]
        namespace = {}
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
        torch.manual_seed(2)
        local = build_xyz_sparse_encoder([32, 48, 41]).cuda().eval()
        reference = namespace['SparseEncoder'](
            in_channels=5, sparse_shape=[32, 48, 41], output_channels=128,
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
            block_type='basicblock').cuda().eval()
        reference.load_state_dict(local.state_dict(), strict=True)
        coords = torch.stack([torch.randint(32, (2000,)), torch.randint(48, (2000,)),
                              torch.randint(40, (2000,))], 1).unique(dim=0).cuda()
        coords = F.pad(coords, (1, 0)).int()
        features = torch.randn(len(coords), 5, device='cuda', requires_grad=True)
        actual = local(features, coords, 1)
        expected = reference(features, coords, 1)
        torch.testing.assert_close(actual, expected)
        grad = torch.autograd.grad(actual.square().mean(), features, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(expected.square().mean(), features)[0]
        torch.testing.assert_close(grad, ref_grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA decoder parity test')
    def test_second_fpn_against_upstream_source(self):
        upstream = Path(os.environ.get('BEVFUSION_SOURCE', '../bevfusion'))
        if not upstream.exists():
            self.skipTest('Set BEVFUSION_SOURCE to the upstream checkout')
        frontend = BEVFusionLidarBackbone(
            point_cloud_range=(-19.2, -32., -5., 19.2, 32., 3.),
            voxel_size=(0.2, 0.2, 0.2)).cuda().eval()
        norm = dict(type='BN', eps=1e-3, momentum=0.01)
        configs = [
            ('backbone', 'backbones/second.py', 'SECOND', dict(
                in_channels=256, out_channels=[128, 256], layer_nums=[5, 5],
                layer_strides=[1, 2], norm_cfg=norm)),
            ('neck', 'necks/second.py', 'SECONDFPN', dict(
                in_channels=[128, 256], out_channels=[256, 256],
                upsample_strides=[1, 2], norm_cfg=norm, use_conv_for_no_stride=True)),
        ]
        reference = torch.nn.ModuleDict()
        for key, relative, name, kwargs in configs:
            path = upstream / 'mmdet3d/models' / relative
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    node.decorator_list = []
            namespace = {}
            exec(compile(tree, str(path), 'exec'), namespace)
            reference[key] = namespace[name](**kwargs).cuda().eval()
        reference.load_state_dict(frontend.decoder.state_dict(), strict=True)
        x = torch.randn(1, 256, 8, 12, device='cuda', requires_grad=True)
        actual = frontend.decoder['neck'](frontend.decoder['backbone'](x))[0]
        expected = reference['neck'](reference['backbone'](x))[0]
        torch.testing.assert_close(actual, expected)
        grad = torch.autograd.grad(actual.square().mean(), x, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(expected.square().mean(), x)[0]
        torch.testing.assert_close(grad, ref_grad)


if __name__ == '__main__':
    unittest.main()
