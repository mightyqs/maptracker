#!/usr/bin/env python3
"""Mapping-only training and sequential LiDAR mini-val pipeline verification.

GT maps are used for training/metrics only. Inference receives points, poses
and scene ordering, never semantic/vector labels or a localization prior.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plugin  # noqa: F401
from plugin.datasets.evaluation.vector_eval import VectorEvaluate


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')


def to_batch(sample):
    return scatter(collate([sample], samples_per_gpu=1), [0])[0]


def infer(model, sample):
    # Construct an explicit whitelist rather than silently forwarding labels.
    batch = to_batch(sample)
    inputs = {k: batch[k] for k in ('points', 'img_metas', 'seq_info')}
    with torch.no_grad():
        result = model(return_loss=False, rescale=False, **inputs)[0]
    assert 'localization' not in result
    for key in ('vectors', 'scores'):
        if not np.isfinite(np.asarray(result[key])).all():
            raise RuntimeError(f'Non-finite mapping output: {key}')
    return result


def controlled_tracking_check(model, dataset):
    """Force nonempty tracks ONLY for interface tests, not reported AP/IoU.

    Frame 0 admits every query. Later frames keep tracks but forbid births;
    this checks ID continuity even when an untrained model scores everything low.
    """
    original = deepcopy(model.mapping_score_thresholds)
    model.mapping_score_thresholds = dict(first=0., track=0., new=1.)
    counts = []
    handle = model.query_propagate.register_forward_hook(
        lambda module, inputs, output: counts.append(len(inputs[0])))
    try:
        scenes = list(dataset.scene_name2idx.values())
        first = scenes[0]
        outputs = [infer(model, dataset[index]) for index in first[:3]]
        ids = outputs[0]['pos_results']['global_ids']
        assert len(ids) == model.head.num_queries
        for output in outputs[1:]:
            np.testing.assert_array_equal(output['pos_results']['global_ids'], ids)
        assert counts == [len(ids), len(ids)], counts
        # Same-scene rewind to frame 0 must clear the old cache as well.
        replay = infer(model, dataset[first[0]])
        np.testing.assert_allclose(replay['vectors'], outputs[0]['vectors'], atol=1e-5)
        try:
            infer(model, dataset[first[2]])
        except ValueError as error:
            assert 'consecutive' in str(error)
        else:
            raise AssertionError('Out-of-order frames were silently accepted')
        second = infer(model, dataset[scenes[1][0]])
        np.testing.assert_array_equal(second['pos_results']['global_ids'], np.arange(len(ids)))
        return dict(propagated_query_counts=counts, ids_preserved=True,
                    scene_reset=True, replay_equal=True, out_of_order_rejected=True,
                    forced_thresholds='first=0, track=0, new=1; validation uses defaults')
    finally:
        handle.remove()
        model.mapping_score_thresholds = original


def update_global_map(storage, result, roi_size):
    """Retain the latest world-frame polyline for each scene-local track ID.

    This is an auditable accumulator, not a geometry-fusion/loop-closure method.
    """
    pos = result['pos_results']
    meta = result['meta']
    scene = storage.setdefault(pos['scene_name'], {})
    raw_vectors = np.asarray(pos['vectors'])
    vectors = raw_vectors.reshape(raw_vectors.shape[0], raw_vectors.shape[1] // 2, 2)
    local = vectors * np.asarray(roi_size) - np.asarray(roi_size) / 2
    rotation = np.asarray(meta['ego2global_rotation'])[:2, :2]
    translation = np.asarray(meta['ego2global_translation'])[:2]
    world = local @ rotation.T + translation
    for index, track_id in enumerate(pos['global_ids']):
        key = str(int(track_id))
        old = scene.get(key, {})
        scene[key] = dict(vector=world[index], label=int(pos['labels'][index]),
                          score=float(pos['scores'][index]),
                          first_frame=old.get('first_frame', pos['local_idx']),
                          last_frame=pos['local_idx'], observations=old.get('observations', 0) + 1)


def save_preview(out, examples, roi_size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 4 * len(examples)), squeeze=False)
    colors = ['tab:orange', 'tab:blue', 'tab:green']
    for row, (scene, pred, gt) in enumerate(examples):
        for label, lines in gt.items():
            for line in lines:
                axes[row, 0].plot(line[:, 0], line[:, 1], color=colors[label], linewidth=1)
        # Show top-20 only for readability. AP uses all emitted predictions.
        order = np.argsort(-np.asarray(pred['scores']))[:20]
        for i in order:
            line = np.asarray(pred['vectors'][i]) * roi_size - np.asarray(roi_size) / 2
            axes[row, 1].plot(line[:, 0], line[:, 1], color=colors[int(pred['labels'][i])], alpha=.6)
        for col, title in enumerate(('GT map', 'Prediction (top 20, pipeline check only)')):
            ax = axes[row, col]
            ax.set(title=f'{scene}: {title}', xlabel='map x / m', ylabel='map y / m',
                   xlim=(-roi_size[0]/2, roi_size[0]/2), ylim=(-roi_size[1]/2, roi_size[1]/2))
            ax.set_aspect('equal')
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'mapping_preview.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='plugin/configs/lidar_mapping/nuscenes_lidar_mapping_mini.py')
    parser.add_argument('--checkpoint')
    parser.add_argument('--train-steps', type=int, default=5)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.train_steps < 0 or (args.train_steps == 0 and not args.checkpoint):
        parser.error('Train at least one step or supply --checkpoint')
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    # The upstream Chamfer metric uses CPU PyTorch. Do not fork its thread
    # pools after CUDA inference; evaluate in-process with bounded CPU threads.
    torch.set_num_threads(2)
    set_random_seed(0)
    cfg = Config.fromfile(args.config)
    cfg.dump(str(out / 'config.py'))
    model = build_model(cfg.model).cuda()
    model.init_weights()
    assert model.localization_head is None and model.lidar_online_mapping
    assert not any('localization_head' in name for name, _ in model.named_parameters())
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.num_iter = 0
    history = []
    torch.cuda.reset_peak_memory_stats()
    if args.train_steps:
        dataset = build_dataset(cfg.data.train)
        assert dataset.localization_prior is None and not dataset.multi_frame
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
        for step in range(args.train_steps):
            batch = to_batch(dataset[step % len(dataset)])
            assert 'img' not in batch and not any(k.startswith('localization') for k in batch)
            optimizer.zero_grad(set_to_none=True)
            loss, logs, _ = model(return_loss=True, **batch)
            assert not any(k.startswith('loc_') for k in logs)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite mapping loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            gradients = {}
            for prefix in ('backbone.encoders', 'backbone.decoder', 'backbone.adapter', 'head', 'seg_decoder'):
                gradients[prefix] = sum(p.grad.abs().sum().item() for name, p in model.named_parameters()
                                        if name.startswith(prefix + '.') and p.grad is not None)
                assert gradients[prefix] > 0, prefix
            optimizer.step()
            model.num_iter += 1
            record = dict(step=step, loss=float(loss), grad_norm=float(grad_norm),
                          gradient_l1=gradients, metrics=logs)
            history.append(record)
            print(json.dumps(record), flush=True)
        save_checkpoint(model, str(out / 'mapping.pth'))
        load_checkpoint(model, str(out / 'mapping.pth'), map_location='cpu', strict=True)
    model.eval()
    val = build_dataset(cfg.data.val)
    assert val.localization_prior is None
    controlled = controlled_tracking_check(model, val)
    print('Controlled tracking checks passed:', controlled, flush=True)

    # Collect GT from this exact val config; avoid the upstream global GT cache
    # whose key does not distinguish mini from the full dataset.
    evaluator = VectorEvaluate(cfg.data.val.eval_config, n_workers=0)
    ground_truth = {}
    for i in range(len(evaluator.dataset)):
        item = evaluator.dataset[i]
        ground_truth[item['img_metas'].data['token']] = item['vectors'].data
    assert set(ground_truth) == set(s['token'] for s in val.samples)
    evaluator.gts = ground_truth
    results, frames, global_map, examples = [], [], {}, []
    intersection, union = np.zeros(3), np.zeros(3)
    for index in range(len(val)):
        sample = val[index]
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = infer(model, sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pos = result['pos_results']
        assert len(np.unique(pos['global_ids'])) == len(pos['global_ids'])
        update_global_map(global_map, result, cfg.model.roi_size)
        target = sample['semantic_mask'].data.numpy()[:, ::-1, :] > 0
        predicted = np.stack([result['semantic_mask'] == c + 1 for c in range(3)])
        intersection += (predicted & target).sum((1, 2))
        union += (predicted | target).sum((1, 2))
        frames.append(dict(token=result['token'], scene=pos['scene_name'],
                           local_idx=pos['local_idx'], detections=len(pos['scores']),
                           propagated=int(np.asarray(result['props']).sum()),
                           active_ids=pos['global_ids'], forward_seconds=elapsed))
        if pos['local_idx'] == 0 and len(examples) < 2:
            examples.append((pos['scene_name'], result, ground_truth[result['token']]))
        results.append(result)
        if (index + 1) % 10 == 0 or index + 1 == len(val):
            print(f'Mapped {index + 1}/{len(val)} sequential frames', flush=True)
    submission = val.format_results(results, prefix=str(out), save_semantic=True)
    write_json(out / 'frames.json', frames)
    write_json(out / 'global_map_latest.json', global_map)
    write_json(out / 'training.json', history)
    write_json(out / 'controlled_tracking.json', controlled)
    save_preview(out, examples, cfg.model.roi_size)
    ap = evaluator.evaluate(submission)
    iou = intersection / np.maximum(union, 1)
    report = dict(config=args.config, initialization=args.checkpoint or 'random',
                  training=history, localization_enabled=False,
                  controlled_tracking=controlled, frames=len(frames), scenes=len(global_map),
                  vector_AP=ap, vector_mAP=float(np.mean(list(ap.values()))),
                  semantic_iou=iou, semantic_miou=float(iou[union > 0].mean()),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                  pose_source='nuScenes GT ego poses and sensor extrinsics',
                  scope='mapping evaluation; online vector-query inference; no long-term memory/BEV fusion',
                  thresholds=model.mapping_score_thresholds)
    write_json(out / 'summary.json', report)
    print(json.dumps(report, default=json_default, indent=2), flush=True)


if __name__ == '__main__':
    main()
