"""LiDAR frontend following MIT-HAN-Lab/BEVFusion (326653d).

Reference: configs/nuscenes/seg/lidar-centerpoint-bev128.yaml and
mmdet3d/models/backbones/sparse_encoder.py. OpenMMLab sparse/SECOND/FPN
implementations are reused from the installed MMDetection3D package.
No imports from a second, conflicting mmdet3d installation are needed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models import BACKBONES
from mmdet3d.models.builder import build_backbone, build_neck


def map_grid_in_lidar(roi_size, bev_h, bev_w, point_cloud_range):
    """MapTracker endpoint grid -> BEVFusion cell-centred native XY BEV.

    Map extractor: (x_map, y_map) = (y_lidar, -x_lidar).
    Upstream dense BEV axes are [X, Y], not [Y, X]. grid_sample's first
    coordinate indexes columns (LiDAR Y); its second indexes rows (LiDAR X).
    Input scope follows BEVFusion BEVGridTransform / align_corners=False.
    Output endpoints follow MapTracker.plane and the localization matcher.
    """
    x = torch.linspace(-roi_size[0] / 2, roi_size[0] / 2, bev_w)
    y = torch.linspace(roi_size[1] / 2, -roi_size[1] / 2, bev_h)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xmin, ymin, _, xmax, ymax, _ = point_cloud_range
    gx = 2 * (xx - ymin) / (ymax - ymin) - 1
    gy = 2 * (-yy - xmin) / (xmax - xmin) - 1
    if (gx.abs() >= 1).any() or (gy.abs() >= 1).any():
        raise ValueError('Native LiDAR range must cover the map ROI with a margin')
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


def build_xyz_sparse_encoder(sparse_shape):
    # These imports stay local: image-only configurations do not build sparse ops.
    from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder
    from mmdet3d.models.middle_encoders.sparse_encoder import SparseConvTensor
    from mmdet3d.ops import make_sparse_convmodule

    class XYZSparseEncoder(SparseEncoder):
        """BEVFusion XYZ layout; layer names match its SparseEncoder.

        The inherited encoder stages are identical to OpenMMLab's implementation.
        Only the height-convolution axis and dense-to-BEV layout differ.
        """
        def __init__(self):
            super().__init__(
                in_channels=5, sparse_shape=sparse_shape, output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64),
                                  (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1),
                                  (0, 0, (1, 1, 0)), (0, 0)),
                block_type='basicblock')
            self.conv_out = make_sparse_convmodule(
                128, 128, kernel_size=(1, 1, 3), stride=(1, 1, 2),
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                padding=0, indice_key='spconv_down2', conv_type='SparseConv3d')

        def forward(self, features, coordinates, batch_size):
            x = SparseConvTensor(features, coordinates.int(), self.sparse_shape,
                                 batch_size)
            x = self.conv_input(x)
            for layer in self.encoder_layers:
                x = layer(x)
            dense = self.conv_out(x).dense()  # B,C,X,Y,Z
            b, c, nx, ny, nz = dense.shape
            return dense.permute(0, 1, 4, 2, 3).contiguous().view(b, c * nz, nx, ny)

    return XYZSparseEncoder()


@BACKBONES.register_module()
class BEVFusionLidarBackbone(nn.Module):
    """Single-sweep, five-channel LiDAR -> MapTracker BEV adapter.

    Inputs are raw sensor-frame [x,y,z,intensity,time_lag]; time_lag is zero
    for a single sweep. Never substitute the nuScenes ring column for time.
    The official 0.1m configuration is the default; a separate mini config
    reduces the XY range/resolution without changing layer widths or depths.
    """
    single_frame_only = True

    def __init__(self, roi_size=(60., 30.), bev_h=50, bev_w=100,
                 point_cloud_range=(-51.2, -51.2, -5., 51.2, 51.2, 3.),
                 voxel_size=(0.1, 0.1, 0.2), max_num_points=10,
                 max_voxels=(90000, 120000), freeze_encoder=False):
        super().__init__()
        from mmcv.ops import Voxelization
        self.point_cloud_range = tuple(point_cloud_range)
        self.voxel_size = tuple(voxel_size)
        self.freeze_encoder = bool(freeze_encoder)
        grid = [(point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i]
                for i in range(3)]
        if any(abs(n - round(n)) > 1e-4 for n in grid):
            raise ValueError('Point-cloud range must be divisible by voxel size')
        nx, ny, nz = [round(n) for n in grid]
        if nx % 16 or ny % 16 or nz != 40:
            raise ValueError('XY grid must be divisible by 16; height needs 40 bins')
        self.encoders = nn.ModuleDict({'lidar': nn.ModuleDict({
            'voxelize': Voxelization(voxel_size=voxel_size,
                                    point_cloud_range=point_cloud_range,
                                    max_num_points=max_num_points,
                                    max_voxels=tuple(max_voxels)),
            'backbone': build_xyz_sparse_encoder([nx, ny, nz + 1]),
        })})
        norm = dict(type='BN', eps=1e-3, momentum=0.01)
        self.decoder = nn.ModuleDict({
            'backbone': build_backbone(dict(
                type='SECOND', in_channels=256, out_channels=[128, 256],
                layer_nums=[5, 5], layer_strides=[1, 2], norm_cfg=norm,
                conv_cfg=dict(type='Conv2d', bias=False))),
            'neck': build_neck(dict(
                type='SECONDFPN', in_channels=[128, 256],
                out_channels=[256, 256], upsample_strides=[1, 2],
                norm_cfg=norm, upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True)),
        })
        self.adapter = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                     nn.GroupNorm(32, 256), nn.SiLU())
        # Reconstructed from configuration, not learned/checkpoint state. MMCV
        # 1.x save_checkpoint also serializes non-persistent buffers, so keep
        # this small constant as a plain tensor and move it at sampling time.
        self.map_grid = map_grid_in_lidar(roi_size, bev_h, bev_w, point_cloud_range)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            for module in (self.encoders, self.decoder):
                module.eval()
                module.requires_grad_(False)
        return self

    @torch.no_grad()
    def voxelize(self, points):
        features, coordinates = [], []
        for batch_index, cloud in enumerate(points):
            if cloud.ndim != 2 or cloud.shape[1] != 5:
                raise ValueError('Expected [N,5] xyz/intensity/time_lag points')
            if not torch.isfinite(cloud).all():
                raise ValueError('LiDAR contains NaN or Inf')
            voxels, zyx, counts = self.encoders['lidar']['voxelize'](
                cloud.float().contiguous())
            if len(counts) == 0:
                raise ValueError(f'No valid LiDAR voxels in batch sample {batch_index}')
            features.append(voxels.sum(1) / counts.to(voxels).unsqueeze(1))
            # MMCV emits ZYX; upstream BEVFusion emits XYZ (including its CUDA op).
            xyz = zyx[:, [2, 1, 0]]
            coordinates.append(F.pad(xyz, (1, 0), value=batch_index))
        return torch.cat(features).contiguous(), torch.cat(coordinates).int()

    def forward(self, img=None, img_metas=None, timestep=0,
                history_bev_feats=None, history_img_metas=None,
                all_history_coord=None, points=None, img_backbone_gradient=True, **kwargs):
        if points is None or len(points) == 0:
            raise ValueError('BEVFusionLidarBackbone requires a nonempty points batch')
        if history_bev_feats or history_img_metas:
            raise ValueError('LiDAR frontend currently supports single-frame input only')
        # Voxelization/sparse ops intentionally run in FP32 for the first baseline.
        if torch.is_autocast_enabled():
            raise ValueError('Validate this LiDAR frontend in FP32; AMP is not enabled')
        features, coordinates = self.voxelize(points)
        # Match upstream's history-frame sensor-encoder gradient policy.
        # BEV decoder/adapter and mapping heads still train on every frame.
        with torch.set_grad_enabled(torch.is_grad_enabled() and img_backbone_gradient):
            native = self.encoders['lidar']['backbone'](features, coordinates, len(points))
        native = self.decoder['neck'](self.decoder['backbone'](native))[0]
        aligned = F.grid_sample(native, self.map_grid.to(native).expand(
            len(points), -1, -1, -1), align_corners=False, padding_mode='zeros')
        return self.adapter(aligned), None
