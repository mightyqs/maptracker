"""Protocol invariants for paired localization evaluation (no GPU needed)."""
import unittest
import numpy as np

from evaluate_fixed import make_manifest, pose_metrics, summarize


class FixedEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.samples = [dict(token=str(i), scene_name='a' if i < 2 else 'b')
                        for i in range(5)]

    def test_manifest_reproducibility_and_cross_scene_donors(self):
        first = make_manifest(self.samples, 245, 5, 123)
        self.assertEqual(first, make_manifest(self.samples, 245, 5, 123))
        self.assertNotEqual(first, make_manifest(self.samples, 245, 5, 124))
        for row in first:
            self.assertNotEqual(row['scene'], row['donor_scene'])
        for token in {r['token'] for r in first}:
            self.assertEqual(len({r['target_index'] for r in first if r['token'] == token}), 5)
        reordered = make_manifest(list(reversed(self.samples)), 245, 5, 123)
        signature = lambda rows: sorted((r['token'], r['repeat'], r['target_index'],
                                         r['donor_token']) for r in rows)
        self.assertEqual(signature(first), signature(reordered))

    def test_no_silent_same_scene_fallback(self):
        with self.assertRaises(ValueError):
            make_manifest(self.samples[:2], 245, 5, 123)

    def test_known_pose_error_and_yaw_wrap(self):
        metrics = pose_metrics([[0, 0, np.deg2rad(179)]],
                               [[3, 4, np.deg2rad(-179)]], 5, 3)
        self.assertEqual(metrics['translation_mean_m'], 5)
        self.assertAlmostEqual(metrics['yaw_mean_deg'], 2)
        self.assertEqual(metrics['success_rate'], 1)
        self.assertEqual(metrics['exact_acc'], 0)

    def test_paired_summary_and_identity_baseline(self):
        correct = dict(pose_map=[1.2, 0, 0], pose_mean=[1.0, 0, 0],
                       soft_nll=3.5, confidence=0.1, normalized_entropy=0.9)
        wrong = dict(correct, pose_map=[0, 0, 0])
        records = [dict(target_pose=[1.2, 0, 0], correct=correct,
                        mismatched=wrong, identity=dict(pose_map=[0, 0, 0]))]
        result = summarize(records, 1, 1)
        self.assertEqual(result['identity']['translation_mean_m'], 1.2)
        self.assertEqual(result['correct']['exact_acc'], 1)
        self.assertEqual(result['mismatched']['exact_acc'], 0)
        self.assertEqual(result['paired']['same_map_prediction_rate'], 0)


if __name__ == '__main__':
    unittest.main()
