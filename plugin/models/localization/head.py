import torch
import torch.nn as nn

from mmdet.models import HEADS

from .core import (
    LocalizationNeck,
    RasterMapEncoder,
    SE2TemplateMatcher,
    SyntheticLocalizationCore,
)


@HEADS.register_module(force=True)
class RasterMapLocalizationHead(nn.Module):
    """Quick nuScenes localization head using rasterized vector-map priors."""

    def __init__(
        self,
        bev_in_channels=256,
        map_in_channels=3,
        hidden_channels=64,
        descriptor_dim=32,
        roi_size=(60.0, 30.0),
        max_translation=3.6,
        translation_step=1.2,
        max_yaw_deg=4.0,
        yaw_step_deg=2.0,
        candidate_chunk_size=32,
        regression_loss_weight=0.25,
        synthetic_train_perturbation=True,
        synthetic_test_perturbation=False,
        detach_bev=False,
        loss_weight=1.0,
    ):
        super().__init__()
        self.synthetic_train_perturbation = bool(synthetic_train_perturbation)
        self.synthetic_test_perturbation = bool(synthetic_test_perturbation)
        self.detach_bev = bool(detach_bev)
        self.loss_weight = float(loss_weight)

        self.localization_neck = LocalizationNeck(
            in_channels=bev_in_channels,
            hidden_channels=hidden_channels,
            descriptor_dim=descriptor_dim,
        )
        self.map_encoder = RasterMapEncoder(
            in_channels=map_in_channels,
            hidden_channels=hidden_channels,
            descriptor_dim=descriptor_dim,
        )
        matcher = SE2TemplateMatcher(
            roi_size=roi_size,
            max_translation=max_translation,
            translation_step=translation_step,
            max_yaw_deg=max_yaw_deg,
            yaw_step_deg=yaw_step_deg,
            candidate_chunk_size=candidate_chunk_size,
        )
        self.core = SyntheticLocalizationCore(
            matcher=matcher,
            regression_loss_weight=regression_loss_weight,
        )

    def forward(
        self,
        bev_features,
        map_raster,
        return_loss=True,
        synthetic_perturbation=None,
        target_indices=None,
    ):
        if self.detach_bev:
            bev_features = bev_features.detach()
        observation = self.localization_neck(bev_features)
        map_features = self.map_encoder(map_raster, output_size=observation.shape[-2:])

        if synthetic_perturbation is None:
            synthetic_perturbation = (
                self.synthetic_train_perturbation
                if return_loss
                else self.synthetic_test_perturbation
            )
        losses, outputs = self.core(
            observation=observation,
            map_features=map_features,
            synthesize_error=synthetic_perturbation,
            target_indices=target_indices,
        )
        if return_loss:
            losses = {
                name: value * self.loss_weight
                for name, value in losses.items()
            }
            return losses, outputs
        return outputs

    @staticmethod
    def result_for_sample(outputs, batch_index):
        result = {}
        for key in (
            'pose_mean',
            'pose_map',
            'covariance',
            'confidence',
            'entropy',
            'normalized_entropy',
            'peak_ratio',
            'target_pose',
        ):
            value = outputs.get(key)
            if value is not None:
                result[key] = value[batch_index].detach().cpu().numpy()
        return result
