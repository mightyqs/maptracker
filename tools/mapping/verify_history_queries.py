#!/usr/bin/env python3
"""Exercise original GT association / query propagation losses on real LiDAR frames.

Default-threshold training and forced-positive gradient diagnostics are reported
separately. The latter never updates or saves model parameters.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from verify_lidar import to_batch, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='plugin/configs/lidar_mapping/nuscenes_lidar_mapping_2frame_mini.py')
    parser.add_argument('--checkpoint')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.pairs < 1:
        parser.error('--steps and --pairs must be positive')
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    set_random_seed(0)
    cfg = Config.fromfile(args.config)
    cfg.dump(str(out / 'config.py'))
    dataset = build_dataset(cfg.data.train)
    assert dataset.multi_frame >= 2 and dataset.matching
    # Avoid repeated scene-start padding in the principal training check.
    indices = [i for ids in dataset.scene_name2idx.values()
               for i in ids[max(dataset.sampling_span, dataset.multi_frame):]][:args.pairs]
    assert len(indices) == args.pairs
    # Fix both frame selection and points for an auditable small-sample test.
    samples = [dataset[i] for i in indices]
    for sample in samples:
        metas = [p['img_metas'].data for p in sample['all_prev_data']] + [sample['img_metas'].data]
        assert len({m['scene_name'] for m in metas}) == 1
        assert all(a['local_idx'] < b['local_idx'] for a, b in zip(metas, metas[1:]))
    model = build_model(cfg.model).cuda()
    model.init_weights()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    assert model.localization_head is None
    model.num_iter = 0
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr,
                                  weight_decay=cfg.optimizer.weight_decay)
    records = []
    encoder_grad_modes, motion_sizes, decoder_history, decoder_totals = [], [], [], []
    handles = [
        model.backbone.encoders['lidar']['backbone'].register_forward_hook(
            lambda m, inp, output: encoder_grad_modes.append(output.requires_grad)),
        model.query_propagate.register_forward_hook(
            lambda m, inp, output: motion_sizes.append(len(inp[0]))),
    ]
    original_head = model.head.forward
    original_transformer = model.head.transformer.forward

    def checked_transformer(*a, **kw):
        decoder_totals.append(kw['query_embed'].shape[1])
        return original_transformer(*a, **kw)

    def checked_head(*a, **kw):
        info = kw.get('track_query_info')
        if info is not None:
            for item in info:
                assert not item['track_query_hs_embeds'].requires_grad
                assert len(item['track_queries_mask']) == len(item['track_query_hs_embeds']) + model.head.num_queries
                decoder_history.append(len(item['track_query_hs_embeds']))
        return original_head(*a, **kw)

    def run(sample, diagnostic=False, update=True):
        encoder_grad_modes.clear()
        motion_sizes.clear()
        decoder_history.clear()
        decoder_totals.clear()
        batch = to_batch(sample)
        assert 'img' not in batch and 'localization_map' not in batch
        optimizer.zero_grad(set_to_none=True)
        with patch.object(model.head, 'forward', side_effect=checked_head), \
                patch.object(model.head.transformer, 'forward', side_effect=checked_transformer):
            loss, logs, _ = model(return_loss=True, **batch)
        assert torch.isfinite(loss)
        assert not any(k.startswith('loc_') for k in logs)
        n = dataset.multi_frame
        assert decoder_totals == [model.head.num_queries] + [
            model.head.num_queries + count for count in decoder_history], decoder_totals
        assert encoder_grad_modes == [False] * (n - 1) + [True], encoder_grad_modes
        for t in range(n - 1):
            assert f'f_trans_t{t}' in logs and f'b_trans_t{t}' in logs
            assert f'cls_t{t}' in logs and f'seg_t{t}' in logs
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        gradients = {prefix: sum(p.grad.abs().sum().item()
            for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None)
            for prefix in ('backbone.encoders', 'backbone.decoder', 'backbone.adapter',
                           'head', 'seg_decoder', 'query_propagate')}
        assert all(value > 0 for key, value in gradients.items() if key != 'query_propagate')
        if diagnostic:
            assert gradients['query_propagate'] > 0
            assert logs['track_matches_t0'] > 0
            assert logs['f_trans_t0'] > 0 and logs['b_trans_t0'] > 0
            assert decoder_history and all(x > 0 for x in decoder_history)
        elif update:
            optimizer.step()
        return dict(loss=loss.item(), grad_norm=float(norm), metrics=logs,
                    gradient_l1=gradients, sparse_encoder_requires_grad=list(encoder_grad_modes),
                    motion_query_counts=list(motion_sizes), decoder_history_counts=list(decoder_history),
                    decoder_total_queries=list(decoder_totals))

    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(args.steps):
            record = run(samples[step % len(samples)])
            records.append(record)
            print(f"step={step} loss={record['loss']:.4f} "
                  f"queries={record['metrics']['track_queries_t0']} "
                  f"matches={record['metrics']['track_matches_t0']} "
                  f"motion_grad={record['gradient_l1']['query_propagate']:.6f}", flush=True)
        # Save ONLY training with original thresholds and losses.
        save_checkpoint(model, str(out / 'mapping.pth'))
        load_checkpoint(model, str(out / 'mapping.pth'), map_location='cpu', strict=True)
        # Random initialization can reject all positive tracks at the original
        # 0.4 threshold. Isolate a forced-positive diagnostic, no optimizer step.
        original_prepare = model.prepare_track_queries_and_targets

        def include_positive_tracks(*a, **kw):
            kw['pos_th'] = 0.0
            return original_prepare(*a, **kw)

        with patch.object(model, 'prepare_track_queries_and_targets', side_effect=include_positive_tracks):
            controlled = run(samples[0], diagnostic=True)
        # Exercise upstream scene-start padding too (real frame 0 repeated).
        load_checkpoint(model, str(out / 'mapping.pth'), map_location='cpu', strict=True)
        boundary = run(dataset[0], update=False)
    finally:
        for handle in handles:
            handle.remove()
    summary = dict(config=args.config, frame_indices=indices, training=records,
        controlled_positive_diagnostic=controlled, scene_start_check=boundary,
        controlled_note='pos_th=0 for diagnostic only; checkpoint uses original pos_th=0.4',
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        localization_enabled=False, history_bev_fusion=False, long_term_memory=False)
    write_json(out / 'summary.json', summary)
    print(f'PASS: {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
