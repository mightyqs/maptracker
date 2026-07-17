#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path


def load_core_module():
    repo_root = Path(__file__).resolve().parents[2]
    core_path = repo_root / 'plugin/models/localization/core.py'
    spec = importlib.util.spec_from_file_location('maptracker_localization_core', core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_finite(name, tensor):
    if not tensor.isfinite().all():
        raise RuntimeError(f'{name} contains non-finite values')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    core = load_core_module()
    torch = core.torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is unavailable; falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    torch.manual_seed(7)

    matcher = core.SE2TemplateMatcher(
        roi_size=(32.0, 16.0),
        max_translation=2.0,
        translation_step=1.0,
        max_yaw_deg=4.0,
        yaw_step_deg=2.0,
        candidate_chunk_size=16,
    ).to(device)
    localization = core.SyntheticLocalizationCore(matcher).to(device)

    observation = torch.nn.functional.interpolate(
        torch.randn(1, 8, 8, 16, device=device),
        size=(16, 32),
        mode='bilinear',
        align_corners=True,
    )
    observation = torch.nn.functional.normalize(observation, dim=1)
    observation.requires_grad_(True)
    desired_pose = torch.tensor(
        [1.0, -1.0, 4.0 * core.math.pi / 180.0],
        device=device,
    )
    target_index = (
        matcher.hypotheses.to(device) - desired_pose
    ).abs().sum(dim=-1).argmin().view(1)
    losses, outputs = localization(
        observation,
        observation,
        synthesize_error=True,
        target_indices=target_index,
    )
    total_loss = sum(losses.values())
    total_loss.backward()

    assert_finite('loss', total_loss)
    assert_finite('covariance', outputs['covariance'])
    if observation.grad is None:
        raise RuntimeError('matcher did not backpropagate to observation features')
    predicted_index = outputs['logits'].argmax(dim=-1)
    print(f'hypotheses: {matcher.num_hypotheses}')
    print(f'target index: {target_index.item()}')
    print(f'predicted index: {predicted_index.item()}')
    print(f'target pose [m, m, rad]: {outputs["target_pose"][0].tolist()}')
    print(f'MAP pose [m, m, rad]: {outputs["pose_map"][0].tolist()}')
    print(f'loss: {total_loss.item():.6f}')
    print(f'confidence: {outputs["confidence"].item():.6f}')
    if predicted_index.item() != target_index.item():
        raise RuntimeError('synthetic SE(2) recovery selected the wrong hypothesis')

    neck = core.LocalizationNeck(
        in_channels=16,
        hidden_channels=16,
        descriptor_dim=8,
    ).to(device)
    map_encoder = core.RasterMapEncoder(
        in_channels=3,
        hidden_channels=16,
        descriptor_dim=8,
    ).to(device)
    bev_input = torch.randn(2, 16, 16, 32, device=device)
    map_input = torch.randn(2, 3, 32, 64, device=device)
    observation_descriptors = neck(bev_input)
    map_descriptors = map_encoder(
        map_input,
        output_size=observation_descriptors.shape[-2:],
    )
    map_decoder = core.SemanticDecoder(
        in_channels=8,
        hidden_channels=16,
        out_channels=3,
    ).to(device)
    bev_decoder = core.SemanticDecoder(
        in_channels=8,
        hidden_channels=16,
        out_channels=3,
    ).to(device)
    semantic_target = (map_input > 0).to(dtype=map_descriptors.dtype)
    map_semantic_logits = map_decoder(
        map_descriptors,
        output_size=semantic_target.shape[-2:],
    )
    bev_semantic_logits = bev_decoder(
        observation_descriptors,
        output_size=semantic_target.shape[-2:],
    )
    encoder_losses, encoder_outputs = localization(
        observation_descriptors,
        map_descriptors,
        synthesize_error=True,
    )
    semantic_losses = (
        core.semantic_focal_loss(map_semantic_logits, semantic_target)
        + core.semantic_dice_loss(map_semantic_logits, semantic_target)
        + core.semantic_focal_loss(bev_semantic_logits, semantic_target)
        + core.semantic_dice_loss(bev_semantic_logits, semantic_target)
    )
    encoder_loss = sum(encoder_losses.values()) + semantic_losses
    encoder_loss.backward()
    assert_finite('encoder loss', encoder_loss)
    assert_finite('encoder covariance', encoder_outputs['covariance'])
    map_iou, map_iou_per_class = core.semantic_iou(
        map_semantic_logits,
        semantic_target,
    )
    assert_finite('map reconstruction IoU', map_iou)
    assert_finite('map reconstruction per-class IoU', map_iou_per_class)
    if map_semantic_logits.shape != semantic_target.shape:
        raise RuntimeError('map decoder output shape does not match its target')
    if bev_semantic_logits.shape != semantic_target.shape:
        raise RuntimeError('BEV decoder output shape does not match its target')
    if not any(parameter.grad is not None for parameter in neck.parameters()):
        raise RuntimeError('localization neck did not receive gradients')
    if not any(parameter.grad is not None for parameter in map_encoder.parameters()):
        raise RuntimeError('map encoder did not receive gradients')
    if not any(parameter.grad is not None for parameter in map_decoder.parameters()):
        raise RuntimeError('map semantic decoder did not receive gradients')
    if not any(parameter.grad is not None for parameter in bev_decoder.parameters()):
        raise RuntimeError('BEV semantic decoder did not receive gradients')
    print(f'encoder output shape: {tuple(observation_descriptors.shape)}')
    print(f'semantic output shape: {tuple(map_semantic_logits.shape)}')
    print('localization smoke test passed')


if __name__ == '__main__':
    main()
