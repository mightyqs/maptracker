#!/usr/bin/env python3
"""Validate and map MIT-HAN-Lab BEVFusion LiDAR-only frontend weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from mmcv import Config
from mmdet3d.models import build_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plugin  # noqa: F401


def map_frontend(source, target):
    groups = ('encoders.lidar.backbone.', 'decoder.backbone.', 'decoder.neck.')
    source = {(k[7:] if k.startswith('module.') else k): v for k, v in source.items()}
    mapped = {}
    errors = []
    for name, expected in target.items():
        if not name.startswith(tuple('backbone.' + group for group in groups)):
            continue
        original = name[len('backbone.'):]
        if original not in source:
            errors.append(f'missing {original}')
        elif source[original].shape != expected.shape:
            errors.append(f'{original}: {tuple(source[original].shape)} != {tuple(expected.shape)}')
        elif not torch.isfinite(source[original]).all():
            errors.append(f'nonfinite {original}')
        else:
            mapped[name] = source[original]
    if errors or not mapped:
        raise ValueError('Frontend mapping failed (no automatic sparse-kernel permutation):\n' + '\n'.join(errors))
    return mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='plugin/configs/lidar_mapping/nuscenes_lidar_mapping_mini.py')
    args = parser.parse_args()
    output = Path(args.output)
    report_path = output.with_suffix('.report.json')
    if output.exists() or report_path.exists():
        parser.error('Output/report already exists')
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model)
    original = torch.load(args.checkpoint, map_location='cpu')
    state = original.get('state_dict', original)
    # A fusion checkpoint can have compatible shapes but a different decoder input.
    if any(k.startswith(('fuser.', 'module.fuser.', 'encoders.camera.',
                         'module.encoders.camera.')) for k in state):
        raise ValueError('Use lidar-only-seg.pth, not a camera/fusion checkpoint')
    mapped = map_frontend(state, model.state_dict())
    incompatible = model.load_state_dict(mapped, strict=False)
    assert not incompatible.unexpected_keys
    for name, value in mapped.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    report = dict(source=str(args.checkpoint), sha256=digest.hexdigest(),
                  tensors=len(mapped), elements=sum(v.numel() for v in mapped.values()),
                  frontend_coverage=1.0, mapped_keys=list(mapped),
                  intentionally_unloaded=list(incompatible.missing_keys),
                  input_contract='xyz/intensity/time_lag; XYZ sparse layout; no axis conversion')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=mapped, meta=report), output)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {len(mapped)} validated tensors to {output}; report: {report_path}')


if __name__ == '__main__':
    main()
