import torch
import torch.nn as nn

from mmdet.models import HEADS

from .core import (
    LocalizationNeck,
    RasterMapEncoder,
    SE2TemplateMatcher,
    SemanticDecoder,
    SyntheticLocalizationCore,
    semantic_dice_loss,
    semantic_focal_loss,
    semantic_iou,
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
        map_reconstruction_loss_weight=0.0,
        bev_semantic_loss_weight=0.0,
        semantic_focal_loss_weight=1.0,
        semantic_dice_loss_weight=1.0,
        semantic_decoder_hidden_channels=None,
    ):
        super().__init__()
        self.synthetic_train_perturbation = bool(synthetic_train_perturbation)
        self.synthetic_test_perturbation = bool(synthetic_test_perturbation)
        self.detach_bev = bool(detach_bev)
        self.loss_weight = float(loss_weight)
        self.map_reconstruction_loss_weight = float(
            map_reconstruction_loss_weight
        )
        self.bev_semantic_loss_weight = float(bev_semantic_loss_weight)
        self.semantic_focal_loss_weight = float(semantic_focal_loss_weight)
        self.semantic_dice_loss_weight = float(semantic_dice_loss_weight)

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
        decoder_hidden_channels = (
            hidden_channels
            if semantic_decoder_hidden_channels is None
            else int(semantic_decoder_hidden_channels)
        )
        self.map_reconstruction_decoder = None
        if self.map_reconstruction_loss_weight > 0:
            self.map_reconstruction_decoder = SemanticDecoder(
                in_channels=descriptor_dim,
                hidden_channels=decoder_hidden_channels,
                out_channels=map_in_channels,
            )
        self.bev_semantic_decoder = None
        if self.bev_semantic_loss_weight > 0:
            self.bev_semantic_decoder = SemanticDecoder(
                in_channels=descriptor_dim,
                hidden_channels=decoder_hidden_channels,
                out_channels=map_in_channels,
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
            semantic_targets = map_raster.to(
                device=observation.device,
                dtype=observation.dtype,
            )
            output_size = semantic_targets.shape[-2:]
            if self.map_reconstruction_decoder is not None:
                map_semantic_logits = self.map_reconstruction_decoder(
                    map_features,
                    output_size=output_size,
                )
                map_focal = semantic_focal_loss(
                    map_semantic_logits,
                    semantic_targets,
                )
                map_dice = semantic_dice_loss(
                    map_semantic_logits,
                    semantic_targets,
                )
                losses['map_recon_focal'] = (
                    self.map_reconstruction_loss_weight
                    * self.semantic_focal_loss_weight
                    * map_focal
                )
                losses['map_recon_dice'] = (
                    self.map_reconstruction_loss_weight
                    * self.semantic_dice_loss_weight
                    * map_dice
                )
                map_iou, map_iou_per_class = semantic_iou(
                    map_semantic_logits,
                    semantic_targets,
                )
                outputs['map_reconstruction_iou'] = map_iou
                outputs['map_reconstruction_iou_per_class'] = map_iou_per_class

            if self.bev_semantic_decoder is not None:
                bev_semantic_logits = self.bev_semantic_decoder(
                    observation,
                    output_size=output_size,
                )
                bev_focal = semantic_focal_loss(
                    bev_semantic_logits,
                    semantic_targets,
                )
                bev_dice = semantic_dice_loss(
                    bev_semantic_logits,
                    semantic_targets,
                )
                losses['bev_sem_focal'] = (
                    self.bev_semantic_loss_weight
                    * self.semantic_focal_loss_weight
                    * bev_focal
                )
                losses['bev_sem_dice'] = (
                    self.bev_semantic_loss_weight
                    * self.semantic_dice_loss_weight
                    * bev_dice
                )
                bev_iou, bev_iou_per_class = semantic_iou(
                    bev_semantic_logits,
                    semantic_targets,
                )
                outputs['bev_semantic_iou'] = bev_iou
                outputs['bev_semantic_iou_per_class'] = bev_iou_per_class

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
