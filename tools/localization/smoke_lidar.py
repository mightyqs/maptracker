#!/usr/bin/env python3
"""Real mini LiDAR: training, checkpoint round-trip, GT-free inference.

This is a pipeline check with random initialization, not an accuracy experiment.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
import plugin  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='plugin/configs/bev_localization/nuscenes_lidar_localization_mini.py')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--mapping-checkpoint', help='Mapping weights for frozen-BEV localization verification')
    parser.add_argument('--out-dir', default='work_dirs/lidar_smoke')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('steps and batch-size must be positive')
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(0)
    cfg = Config.fromfile(args.config)
    cfg.dump(str(out / 'config.py'))
    dataset = build_dataset(cfg.data.train)
    model = build_model(cfg.model).cuda()
    model.init_weights()
    if args.mapping_checkpoint:
        if not model.freeze_mapping_for_localization:
            parser.error('--mapping-checkpoint requires a frozen mapping localization config')
        source = torch.load(args.mapping_checkpoint, map_location='cpu')['state_dict']
        missing, unexpected = model.load_state_dict(source, strict=False)
        if unexpected or any(not name.startswith('localization_head.') for name in missing):
            raise RuntimeError(f'Unexpected mapping checkpoint mismatch: {missing}, {unexpected}')
    model.num_iter = 0
    model.train()
    frozen = model.freeze_mapping_for_localization
    frozen_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                    if frozen and not name.startswith('localization_head.')}
    if frozen:
        assert all(name.startswith('localization_head.') for name, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    metrics = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        samples = [dataset[(step * args.batch_size + j) % len(dataset)]
                   for j in range(args.batch_size)]
        batch = scatter(collate(samples, samples_per_gpu=args.batch_size), [0])[0]
        assert 'img' not in batch
        optimizer.zero_grad(set_to_none=True)
        loss, logs, count = model(return_loss=True, **batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        groups = {}
        for prefix in ('backbone.encoders', 'backbone.decoder', 'backbone.adapter',
                       'localization_head.localization_neck', 'head', 'seg_decoder'):
            total = sum(p.grad.detach().abs().sum().item() for name, p in model.named_parameters()
                        if name.startswith(prefix + '.') and p.grad is not None)
            groups[prefix] = total
        required = ('localization_head.localization_neck',) if frozen else (
            'backbone.encoders', 'backbone.decoder', 'backbone.adapter',
            'localization_head.localization_neck')
        for prefix in required:
            if groups[prefix] <= 0:
                raise RuntimeError(f'Missing gradient: {prefix}')
        if not model.skip_vector_head and (groups['head'] <= 0 or groups['seg_decoder'] <= 0):
            raise RuntimeError('Joint heads did not receive gradients')
        optimizer.step()
        if frozen:
            for name, value in frozen_state.items():
                if not torch.equal(value, model.state_dict()[name].detach().cpu()):
                    raise RuntimeError(f'Frozen mapping state changed: {name}')
        model.num_iter += 1
        row = dict(step=step, loss=float(loss), grad_norm=float(norm),
                   batch_size=count, gradient_l1=groups, metrics=logs)
        metrics.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    # Real validation sample, with native donor geometry and no loaded images.
    val_dataset = build_dataset(cfg.data.val)
    # Start from a non-first scene frame: single-frame mode must not require
    # an earlier frame's history or vector-query cache.
    sample = val_dataset[1]
    val = scatter(collate([sample], samples_per_gpu=1), [0])[0]
    with torch.no_grad():
        bev = model.extract_observation_bev(img_metas=val['img_metas'], points=val['points'])
        assert tuple(bev.shape) == (1, 256, 50, 100)
        labeled = model(return_loss=False, rescale=False, **val)[0]
        val.pop('semantic_mask')
        val.pop('localization_target_pose')
        unlabeled = model(return_loss=False, rescale=False, **val)[0]
        for key, value in unlabeled['localization'].items():
            np.testing.assert_allclose(labeled['localization'][key], value, atol=1e-6)
    checkpoint = out / 'smoke.pth'
    save_checkpoint(model, str(checkpoint))
    # Perturb one learned tensor to ensure loading actually restores parameters.
    with torch.no_grad():
        next(model.backbone.adapter.parameters()).add_(1.)
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
    model.eval()
    with torch.no_grad():
        restored = model.extract_observation_bev(img_metas=val['img_metas'], points=val['points'])
        torch.testing.assert_close(restored, bev)
    report = dict(config=args.config, steps=metrics, bev_shape=list(bev.shape),
                  frozen_mapping_state_unchanged=frozen, mapping_checkpoint=args.mapping_checkpoint,
                  peak_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
                  gt_free_inference_equal=True, checkpoint_roundtrip=True,
                  localization={key: np.asarray(value).tolist()
                                for key, value in unlabeled['localization'].items()})
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
