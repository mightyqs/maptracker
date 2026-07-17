import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def meshgrid_ij(*tensors):
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # MapTracker's reference environment uses PyTorch 1.9.
        return torch.meshgrid(*tensors)


def _group_count(channels):
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class LocalizationNeck(nn.Module):
    """Project MapTracker BEV features into a compact metric descriptor."""

    def __init__(self, in_channels=256, hidden_channels=64, descriptor_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, stride=2),
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, descriptor_dim, kernel_size=1, bias=False),
        )

    def forward(self, features):
        return F.normalize(self.layers(features), dim=1, eps=1e-6)


class RasterMapEncoder(nn.Module):
    """Encode a high-resolution semantic raster at the localization scale."""

    def __init__(self, in_channels=3, hidden_channels=64, descriptor_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels // 2, stride=2),
            ConvNormAct(hidden_channels // 2, hidden_channels, stride=2),
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, descriptor_dim, kernel_size=1, bias=False),
        )

    def forward(self, raster, output_size=None):
        descriptors = self.layers(raster.float())
        if output_size is not None and descriptors.shape[-2:] != output_size:
            descriptors = F.interpolate(
                descriptors,
                size=output_size,
                mode='bilinear',
                align_corners=True,
            )
        return F.normalize(descriptors, dim=1, eps=1e-6)


def invert_se2(poses):
    """Invert poses stored as forward-x, left-y, counter-clockwise-yaw."""
    dx, dy, yaw = poses.unbind(dim=-1)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    inv_x = -cos_yaw * dx - sin_yaw * dy
    inv_y = sin_yaw * dx - cos_yaw * dy
    return torch.stack((inv_x, inv_y, -yaw), dim=-1)


def build_sampling_grid(poses, height, width, roi_size, dtype=None):
    """Build output-to-input sampling grids in metric BEV coordinates."""
    if poses.ndim != 2 or poses.shape[-1] != 3:
        raise ValueError('poses must have shape [B, 3]')

    dtype = dtype or poses.dtype
    device = poses.device
    roi_x, roi_y = float(roi_size[0]), float(roi_size[1])
    xs = torch.linspace(-roi_x / 2, roi_x / 2, width, device=device, dtype=dtype)
    ys = torch.linspace(roi_y / 2, -roi_y / 2, height, device=device, dtype=dtype)
    grid_y, grid_x = meshgrid_ij(ys, xs)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    dx = poses[:, 0].to(dtype=dtype).view(-1, 1, 1)
    dy = poses[:, 1].to(dtype=dtype).view(-1, 1, 1)
    yaw = poses[:, 2].to(dtype=dtype).view(-1, 1, 1)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    source_x = cos_yaw * grid_x - sin_yaw * grid_y + dx
    source_y = sin_yaw * grid_x + cos_yaw * grid_y + dy
    normalized_x = 2.0 * source_x / roi_x
    normalized_y = -2.0 * source_y / roi_y
    return torch.stack((normalized_x, normalized_y), dim=-1)


def warp_bev(features, poses, roi_size, mode='bilinear'):
    """Apply an SE(2) output-to-input sampling transform to BEV features."""
    height, width = features.shape[-2:]
    grid = build_sampling_grid(
        poses,
        height,
        width,
        roi_size,
        dtype=features.dtype,
    )
    warped = F.grid_sample(
        features,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=True,
    )
    valid = ((grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0))
    return warped, valid.unsqueeze(1).to(dtype=features.dtype)


def symmetric_values(max_abs, step):
    if max_abs < 0 or step <= 0:
        raise ValueError('max_abs must be non-negative and step must be positive')
    count = int(math.floor(max_abs / step + 1e-6))
    return torch.arange(-count, count + 1, dtype=torch.float32) * float(step)


class SE2TemplateMatcher(nn.Module):
    """Exhaustive local SE(2) matcher that returns a joint pose distribution."""

    def __init__(
        self,
        roi_size=(60.0, 30.0),
        max_translation=3.6,
        translation_step=1.2,
        max_yaw_deg=4.0,
        yaw_step_deg=2.0,
        candidate_chunk_size=32,
        initial_logit_scale=10.0,
    ):
        super().__init__()
        self.roi_size = tuple(float(v) for v in roi_size)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.translation_step = float(translation_step)
        self.yaw_step = math.radians(float(yaw_step_deg))

        translations = symmetric_values(max_translation, translation_step)
        yaws = symmetric_values(max_yaw_deg, yaw_step_deg) * (math.pi / 180.0)
        yaw_grid, dy_grid, dx_grid = meshgrid_ij(
            yaws,
            translations,
            translations,
        )
        hypotheses = torch.stack(
            (dx_grid.flatten(), dy_grid.flatten(), yaw_grid.flatten()),
            dim=-1,
        )
        self.register_buffer('hypotheses', hypotheses, persistent=True)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_logit_scale)))

    @property
    def num_hypotheses(self):
        return int(self.hypotheses.shape[0])

    def _score_chunk(self, observation, map_features, candidates):
        batch_size, channels, height, width = observation.shape
        num_candidates = candidates.shape[0]
        map_batch = map_features[:, None].expand(
            batch_size,
            num_candidates,
            channels,
            height,
            width,
        ).reshape(batch_size * num_candidates, channels, height, width)
        pose_batch = candidates[None].expand(batch_size, num_candidates, 3).reshape(-1, 3)
        warped_map, valid = warp_bev(map_batch, pose_batch, self.roi_size)
        observation_batch = observation[:, None].expand(
            batch_size,
            num_candidates,
            channels,
            height,
            width,
        ).reshape_as(warped_map)

        similarity = (observation_batch * warped_map).sum(dim=1, keepdim=True)
        score = (similarity * valid).sum(dim=(1, 2, 3))
        score = score / valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return score.view(batch_size, num_candidates)

    def forward(self, observation, map_features):
        if observation.shape != map_features.shape:
            raise ValueError(
                'observation and map features must have the same shape, got '
                f'{tuple(observation.shape)} and {tuple(map_features.shape)}'
            )

        scores = []
        for start in range(0, self.num_hypotheses, self.candidate_chunk_size):
            candidates = self.hypotheses[start:start + self.candidate_chunk_size]
            scores.append(self._score_chunk(observation, map_features, candidates))
        scores = torch.cat(scores, dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scores * scale

    def decode(self, logits):
        probabilities = logits.softmax(dim=-1)
        hypotheses = self.hypotheses.to(dtype=logits.dtype)

        mean_xy = probabilities @ hypotheses[:, :2]
        mean_sin = probabilities @ torch.sin(hypotheses[:, 2])
        mean_cos = probabilities @ torch.cos(hypotheses[:, 2])
        mean_yaw = torch.atan2(mean_sin, mean_cos).unsqueeze(-1)
        mean_pose = torch.cat((mean_xy, mean_yaw), dim=-1)

        raw_differences = hypotheses.unsqueeze(0) - mean_pose.unsqueeze(1)
        yaw_difference = torch.atan2(
            torch.sin(raw_differences[..., 2]),
            torch.cos(raw_differences[..., 2]),
        )
        differences = torch.cat(
            (raw_differences[..., :2], yaw_difference.unsqueeze(-1)),
            dim=-1,
        )
        covariance = torch.einsum(
            'bn,bni,bnj->bij',
            probabilities,
            differences,
            differences,
        )

        map_indices = logits.argmax(dim=-1)
        map_pose = hypotheses[map_indices]
        top_values = probabilities.topk(k=min(2, self.num_hypotheses), dim=-1).values
        if top_values.shape[-1] == 1:
            peak_ratio = torch.ones_like(top_values[:, 0])
        else:
            peak_ratio = top_values[:, 0] / top_values[:, 1].clamp_min(1e-8)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(max(self.num_hypotheses, 2))

        return {
            'logits': logits,
            'probabilities': probabilities,
            'pose_mean': mean_pose,
            'pose_map': map_pose,
            'covariance': covariance,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'peak_ratio': peak_ratio,
            'confidence': (1.0 - normalized_entropy).clamp(0.0, 1.0),
        }


class SyntheticLocalizationCore(nn.Module):
    """Create exact synthetic prior errors and train the local pose matcher."""

    def __init__(
        self,
        matcher,
        regression_loss_weight=0.25,
        label_sigma_bins=0.75,
    ):
        super().__init__()
        self.matcher = matcher
        self.regression_loss_weight = float(regression_loss_weight)
        self.label_sigma_bins = float(label_sigma_bins)

    def synthesize_prior_error(self, aligned_map, target_indices=None):
        batch_size = aligned_map.shape[0]
        if target_indices is None:
            target_indices = torch.randint(
                self.matcher.num_hypotheses,
                size=(batch_size,),
                device=aligned_map.device,
            )
        target_pose = self.matcher.hypotheses[target_indices].to(
            device=aligned_map.device,
            dtype=aligned_map.dtype,
        )
        inverse_pose = invert_se2(target_pose)
        misaligned_map, _ = warp_bev(aligned_map, inverse_pose, self.matcher.roi_size)
        return misaligned_map, target_indices, target_pose

    def forward(
        self,
        observation,
        map_features,
        synthesize_error=False,
        target_indices=None,
    ):
        target_pose = None
        if synthesize_error:
            map_features, target_indices, target_pose = self.synthesize_prior_error(
                map_features,
                target_indices,
            )

        logits = self.matcher(observation, map_features)
        outputs = self.matcher.decode(logits)
        outputs['target_indices'] = target_indices
        outputs['target_pose'] = target_pose

        losses = {}
        if target_indices is not None:
            hypothesis_error = self.matcher.hypotheses.to(
                device=logits.device,
                dtype=logits.dtype,
            ).unsqueeze(0) - target_pose.unsqueeze(1)
            yaw_hypothesis_error = torch.atan2(
                torch.sin(hypothesis_error[..., 2]),
                torch.cos(hypothesis_error[..., 2]),
            )
            hypothesis_error = torch.cat(
                (hypothesis_error[..., :2], yaw_hypothesis_error.unsqueeze(-1)),
                dim=-1,
            )
            bin_scale = logits.new_tensor([
                self.matcher.translation_step,
                self.matcher.translation_step,
                self.matcher.yaw_step,
            ])
            squared_bin_error = (hypothesis_error / bin_scale).square().sum(dim=-1)
            soft_targets = torch.softmax(
                -0.5 * squared_bin_error / (self.label_sigma_bins ** 2),
                dim=-1,
            )
            losses['loc_nll'] = -(
                soft_targets * F.log_softmax(logits, dim=-1)
            ).sum(dim=-1).mean()
            raw_pose_error = outputs['pose_mean'] - target_pose
            yaw_error = torch.atan2(
                torch.sin(raw_pose_error[:, 2]),
                torch.cos(raw_pose_error[:, 2]),
            )
            pose_error = torch.cat(
                (raw_pose_error[:, :2], yaw_error.unsqueeze(-1)),
                dim=-1,
            )
            scale = self.matcher.hypotheses.abs().amax(dim=0).to(pose_error.dtype)
            scale = scale.clamp_min(1e-3)
            losses['loc_reg'] = self.regression_loss_weight * F.smooth_l1_loss(
                pose_error / scale,
                torch.zeros_like(pose_error),
            )
            outputs['exact_accuracy'] = (
                logits.argmax(dim=-1) == target_indices
            ).float().mean()
        return losses, outputs
