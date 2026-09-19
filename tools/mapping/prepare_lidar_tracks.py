#!/usr/bin/env python3
"""Generate LiDAR-frame GT tracks using the original MapTracker IoU matcher."""
import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools' / 'tracking'))
import plugin  # noqa: F401
from prepare_gt_tracks import form_gt_track_single


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='plugin/configs/lidar_mapping/nuscenes_lidar_mapping_2frame_mini.py')
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = Config.fromfile(args.config)
    data_cfg = deepcopy(cfg.data[args.split])
    output = Path(data_cfg.get('matching_file') or
                  (str(data_cfg.ann_file)[:-4] + '_lidar_gt_tracks.pkl'))
    if output.exists():
        raise FileExistsError(f'{output} already exists; keep it or explicitly remove it before regenerating')
    data_cfg.update(multi_frame=False, matching=False, localization_prior=None)
    # Use identical vectorization/order to training, without cyclic permutations.
    # The LiDAR loader also supplies the map-frame poses used during training.
    vectorizer = deepcopy(next(p for p in cfg.train_pipeline if p.type == 'VectorizeMap'))
    vectorizer.update(permute=False)
    data_cfg.pipeline = [vectorizer,
        dict(type='LoadNuScenesLidarForBEVFusion'),
        dict(type='FormatBundleMap', process_img=False),
        dict(type='Collect3D', keys=['vectors'], meta_keys=(
            'token', 'ego2global_translation', 'ego2global_rotation', 'scene_name'))]
    dataset = build_dataset(data_cfg)
    tracks = {}
    for scene, indices in dataset.scene_name2idx.items():
        _, info = form_gt_track_single(scene, dataset.scene_name2idx, dataset,
                                      None, cfg, SimpleNamespace(visualize=False))
        info['sample_tokens'] = [dataset.samples[i]['token'] for i in indices]
        info['coordinate_frame'] = 'lidar_planar_map_xy'
        # Verify the generated IDs address every vector, in exactly this order.
        for index, ids in zip(indices, info['instance_ids']):
            vectors = dataset[index]['vectors'].data
            for label, lines in vectors.items():
                assert set(ids[label]) == set(range(len(lines)))
        tracks[scene] = info
    # Exclusive creation prevents overwriting an existing camera or LiDAR cache.
    with output.open('xb') as stream:
        pickle.dump(tracks, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(dataset)} frames / {len(tracks)} scenes to {output}', flush=True)


if __name__ == '__main__':
    main()
