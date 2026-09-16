#!/usr/bin/env python3
"""Paired, fixed-perturbation evaluation of the single-frame localization head.

Wrong images are complete six-camera observations from a different scene,
encoded with their OWN camera geometry. Only the cached observation descriptor
is swapped; map, perturbation, and localization target are unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def stable_index(key, count):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % count


def make_manifest(samples, num_hypotheses, repeats, seed):
    if repeats < 1 or repeats > num_hypotheses:
        raise ValueError('repeats must be between 1 and num_hypotheses')
    if len({s['scene_name'] for s in samples}) < 2:
        raise ValueError('Cross-scene mismatches require at least two scenes')
    records = []
    for index, sample in enumerate(samples):
        donors = sorted(
            (i for i, other in enumerate(samples)
             if other['scene_name'] != sample['scene_name']),
            key=lambda i: samples[i]['token'],
        )
        # Hash ordering samples without replacement, independent of input order.
        targets = sorted(range(num_hypotheses), key=lambda h: hashlib.sha256(
            f'{seed}:target:{sample["token"]}:{h}'.encode()).digest())[:repeats]
        for repeat, target in enumerate(targets):
            donor = donors[stable_index(
                f'{seed}:donor:{sample["token"]}:{repeat}', len(donors))]
            records.append(dict(index=index, token=sample['token'],
                                scene=sample['scene_name'], repeat=repeat,
                                target_index=target, donor_index=donor,
                                donor_token=samples[donor]['token'],
                                donor_scene=samples[donor]['scene_name']))
    return records


def pose_metrics(prediction, target, translation_threshold, yaw_threshold):
    prediction, target = np.asarray(prediction), np.asarray(target)
    difference = prediction - target
    yaw = np.abs(np.arctan2(np.sin(difference[:, 2]),
                           np.cos(difference[:, 2]))) * 180 / math.pi
    translation = np.linalg.norm(difference[:, :2], axis=1)
    return dict(
        count=len(target), x_mae_m=float(np.abs(difference[:, 0]).mean()),
        y_mae_m=float(np.abs(difference[:, 1]).mean()),
        translation_mean_m=float(translation.mean()),
        translation_median_m=float(np.median(translation)),
        translation_p95_m=float(np.percentile(translation, 95)),
        yaw_mean_deg=float(yaw.mean()), yaw_p95_deg=float(np.percentile(yaw, 95)),
        exact_acc=float(((translation < 1e-5) & (yaw < 1e-4)).mean()),
        success_rate=float(((translation <= translation_threshold) &
                            (yaw <= yaw_threshold)).mean()),
    )


def summarize(records, translation_threshold, yaw_threshold):
    target = np.asarray([r['target_pose'] for r in records])
    result = {}
    for condition in ('identity', 'correct', 'mismatched'):
        poses = np.asarray([r[condition]['pose_map'] for r in records])
        result[condition] = pose_metrics(poses, target, translation_threshold, yaw_threshold)
        if condition != 'identity':
            result[condition]['mean_pose_metrics'] = pose_metrics(
                [r[condition]['pose_mean'] for r in records], target,
                translation_threshold, yaw_threshold)
            for key in ('soft_nll', 'confidence', 'normalized_entropy'):
                result[condition][key] = float(np.mean([r[condition][key] for r in records]))
    correct = np.asarray([r['correct']['pose_map'] for r in records])
    wrong = np.asarray([r['mismatched']['pose_map'] for r in records])
    result['paired'] = dict(
        same_map_prediction_rate=float(np.isclose(correct, wrong, atol=1e-6).all(1).mean()),
        correct_minus_mismatched_exact_acc=(result['correct']['exact_acc'] -
                                           result['mismatched']['exact_acc']),
        mismatched_minus_correct_translation_m=(result['mismatched']['translation_mean_m'] -
                                               result['correct']['translation_mean_m']),
    )
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--map-source', choices=['auto', 'feature', 'real'], default='auto',
                        help='auto uses real queries when dataset config enables localization_prior')
    parser.add_argument('--translation-threshold', type=float, default=1.0)
    parser.add_argument('--yaw-threshold', type=float, default=1.0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error('--out-dir must be empty to preserve earlier experiments')
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    import plugin  # noqa: F401: register project modules

    set_random_seed(args.seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(args.config)
    map_source = args.map_source
    if map_source == 'auto':
        map_source = 'real' if cfg.data.val.get('localization_prior') else 'feature'
    if map_source == 'feature' and cfg.model.localization_cfg.get('require_prior_map', False):
        raise ValueError('Use the legacy config for feature-level perturbation comparison')
    if (cfg.model.get('use_memory', False) or cfg.model.get('history_steps', 0) != 0
            or cfg.model.get('test_time_history_steps', 0) != 0):
        raise ValueError('This evaluator supports single-frame, no-memory configs only')
    if cfg.get('fp16') is not None:
        raise ValueError('This evaluator currently uses FP32 only')
    cfg.model.pretrained = None
    cfg.model.backbone_cfg.img_backbone.pretrained = None
    dataset_cfg = cfg.data.val.copy()
    dataset_cfg.update(test_mode=True, multi_frame=False, matching=False)
    dataset = build_dataset(dataset_cfg)
    import pickle
    with open(cfg.data.train.ann_file, 'rb') as stream:
        train_samples = pickle.load(stream)
    overlap = sorted({s['scene_name'] for s in train_samples} &
                     {s['scene_name'] for s in dataset.samples})
    if overlap:
        raise ValueError(f'Train/val scene overlap: {overlap}')
    cfg.dump(str(out_dir / 'resolved_config.py'))
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    head = model.localization_head
    if head is None:
        raise ValueError('Config requires a localization head')
    hypotheses = head.core.matcher.hypotheses.clone()
    manifest = make_manifest(dataset.samples, len(hypotheses), args.repeats, args.seed)
    for row in manifest:
        row['target_pose'] = hypotheses[row['target_index']].tolist()
    prior_rasters = []
    if map_source == 'real':
        from plugin.datasets.pipelines.rasterize import RasterizeMap
        raster_cfg = next(step.copy() for step in dataset_cfg.pipeline
                          if step['type'] == 'RasterizeMap' and
                          step.get('output_key', 'semantic_mask') == 'semantic_mask')
        raster_cfg.pop('type')
        rasterizer = RasterizeMap(**raster_cfg)
        for index, row in enumerate(manifest):
            prior = dataset.get_localization_prior(row['index'], row['target_pose'])
            prior_rasters.append(rasterizer.get_semantic_mask(prior['localization_map_geoms']))
            row['prior_global_pose'] = prior['localization_prior_global_pose']
            row['observation_global_pose'] = prior['localization_observation_global_pose']
            if (index + 1) % 50 == 0:
                print(f'Queried {index + 1}/{len(manifest)} real map crops', flush=True)
        np.savez_compressed(out_dir / 'prior_rasters.npz', rasters=np.stack(prior_rasters))
    write_json(out_dir / 'manifest.json', dict(seed=args.seed, repeats=args.repeats,
               hypotheses=hypotheses.tolist(), samples=len(dataset), trials=manifest))
    report = dict(config=str(Path(args.config).resolve()), seed=args.seed,
                  samples=len(dataset), trials=len(manifest), train_val_scene_overlap=overlap,
                  translation_threshold_m=args.translation_threshold,
                  yaw_threshold_deg=args.yaw_threshold,
                  protocol=f'{map_source} map SE2; cross-scene observation swap with donor geometry',
                  semantic_protocol='aligned-map reconstruction and aligned BEV diagnostics',
                  checkpoints=[])
    model.cuda().eval()
    for checkpoint_index, checkpoint in enumerate(args.checkpoints):
        print(f'Loading {checkpoint}', flush=True)
        load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
        if not torch.equal(head.core.matcher.hypotheses.cpu(), hypotheses):
            raise ValueError('Checkpoint hypothesis grid differs from the fixed manifest')
        model.eval()
        cache = []
        semantic_counts = {}
        with torch.no_grad():
            for index in range(len(dataset)):
                sample = dataset[index]
                image = sample['img'].data.unsqueeze(0).cuda()
                metas = [sample['img_metas'].data]
                # Identical no-history backbone + neck path and raster orientation
                # to MapTracker.forward_test; skip unused map/vector decoders.
                bev, _ = model.backbone(image, metas, 0, [], [], [], points=None)
                bev = model.neck(bev)
                raster = sample['semantic_mask'].data.unsqueeze(0).cuda().flip(2)
                observation = head.localization_neck(bev)
                map_features = head.map_encoder(raster, output_size=observation.shape[-2:])
                for name, decoder, features in (
                    ('map_reconstruction', head.map_reconstruction_decoder, map_features),
                    ('bev_semantic', head.bev_semantic_decoder, observation),
                ):
                    if decoder is not None:
                        prediction = decoder(features, raster.shape[-2:]).sigmoid() >= 0.5
                        target = raster >= 0.5
                        intersection = (prediction & target).sum((0, 2, 3)).cpu().numpy()
                        union = (prediction | target).sum((0, 2, 3)).cpu().numpy()
                        counts = semantic_counts.setdefault(name, np.zeros((2, len(intersection))))
                        counts += np.stack((intersection, union))
                if index == 0:
                    target_indices = torch.tensor([manifest[0]['target_index']], device='cuda')
                    check_raster, check_features = raster, map_features
                    target_pose = None
                    if map_source == 'real':
                        check_raster = torch.from_numpy(prior_rasters[0]).unsqueeze(0).cuda().flip(2)
                        check_features = head.map_encoder(check_raster, output_size=observation.shape[-2:])
                        target_pose = hypotheses[target_indices.cpu()].cuda()
                    direct = head(bev, check_raster, return_loss=False,
                                  synthetic_perturbation=(map_source == 'feature'),
                                  target_indices=target_indices, target_pose=target_pose)
                    _, cached = head.core(observation, check_features,
                                          map_source == 'feature', target_indices, target_pose)
                    torch.testing.assert_close(cached['logits'], direct['logits'])
                cache.append((observation.cpu(), map_features.cpu()))
                if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                    print(f'Encoded {index + 1}/{len(dataset)} frames', flush=True)
            records = []
            for trial_index, trial in enumerate(manifest):
                row = dict(trial)
                row['identity'] = dict(pose_map=[0.0, 0.0, 0.0])
                target_indices = torch.tensor([trial['target_index']], device='cuda')
                map_features = cache[trial['index']][1].cuda()
                target_pose = None
                if map_source == 'real':
                    raster = torch.from_numpy(prior_rasters[trial_index]).unsqueeze(0).cuda().flip(2)
                    map_features = head.map_encoder(raster, output_size=cache[trial['index']][0].shape[-2:])
                    target_pose = hypotheses[target_indices.cpu()].cuda()
                for condition, source in (('correct', trial['index']),
                                          ('mismatched', trial['donor_index'])):
                    losses, output = head.core(cache[source][0].cuda(), map_features,
                                               synthesize_error=(map_source == 'feature'),
                                               target_indices=target_indices, target_pose=target_pose)
                    if not torch.isfinite(output['logits']).all():
                        raise RuntimeError(f'Non-finite logits: {trial}')
                    row[condition] = dict(
                        pose_map=output['pose_map'][0].cpu().tolist(),
                        pose_mean=output['pose_mean'][0].cpu().tolist(),
                        soft_nll=float(losses['loc_nll']),
                        confidence=float(output['confidence'][0]),
                        normalized_entropy=float(output['normalized_entropy'][0]))
                records.append(row)
                if (trial_index + 1) % 100 == 0:
                    print(f'Matched {trial_index + 1}/{len(manifest)} paired trials', flush=True)
        metrics = summarize(records, args.translation_threshold, args.yaw_threshold)
        scene_metrics = {scene: summarize([r for r in records if r['scene'] == scene],
                          args.translation_threshold, args.yaw_threshold)
                         for scene in sorted({r['scene'] for r in records})}
        semantic_metrics = {}
        for name, counts in semantic_counts.items():
            iou = counts[0] / np.maximum(counts[1], 1)
            semantic_metrics[name] = dict(per_class_iou=iou.tolist(),
                miou=float(iou[counts[1] > 0].mean()) if (counts[1] > 0).any() else 1.0)
        result = dict(checkpoint=str(Path(checkpoint).resolve()), metrics=metrics,
                      scenes=scene_metrics, semantic=semantic_metrics)
        prefix = f'{checkpoint_index:02d}_{Path(checkpoint).stem}'
        write_json(out_dir / f'{prefix}_records.json', records)
        write_json(out_dir / f'{prefix}_summary.json', result)
        report['checkpoints'].append(result)
        write_json(out_dir / 'summary.json', report)
        print(json.dumps(result, indent=2), flush=True)
    print(f'Results saved to {out_dir}', flush=True)


if __name__ == '__main__':
    main()
