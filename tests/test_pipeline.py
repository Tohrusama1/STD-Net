"""Regression checks for leakage, chronology, graph semantics and scoring."""
import unittest
import numpy as np
import pandas as pd
import torch
from stdnet.preprocessing import FlowPreprocessor, binary_labels, information_value
from stdnet.graph import build_graphs
from stdnet.model import STDNet
from stdnet.losses import HeteroscedasticFocalLoss
from stdnet.metrics import classification_metrics
from stdnet.prepare import split_flows


def flows():
    return pd.DataFrame({
        'Src IP': ['a'] * 4 + ['b'] * 4,
        'Dst IP': ['c'] * 8,
        'Flow ID': ['f1'] * 4 + ['f2'] * 4,
        'Timestamp': ['2025-01-01 00:00:02', '2025-01-01 00:00:01',
                      '2025-01-01 00:00:01', '2025-01-01 00:00:03'] * 2,
        'Bytes': [2, 1, 9, 3, 20, 10, 90, 30],
        'Label': ['Normal', 'Normal', 'DoS', 'DoS'] * 2,
    })


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.frame = flows()
        self.processor = FlowPreprocessor(iv_min=0, grouping='pair').fit(self.frame)

    def test_labels_cannot_change_test_features_or_users(self):
        base, x = self.processor.transform(self.frame)
        flipped = self.frame.copy()
        flipped['Label'] = ['DoS'] * len(flipped)
        other_base, other_x = self.processor.transform(flipped)
        np.testing.assert_array_equal(x, other_x)
        self.assertEqual(base['_user'].tolist(), other_base['_user'].tolist())
        self.assertFalse(any('label' in c.lower() for c in self.processor.feature_columns_))

    def test_training_metadata_frozen(self):
        features = self.processor.feature_columns_.copy()
        mean = self.processor.scaler_.mean_.copy()
        iv = self.processor.iv_.copy()
        self.processor.transform(self.frame.iloc[:4])
        self.assertEqual(features, self.processor.feature_columns_)
        self.assertEqual(iv, self.processor.iv_)
        np.testing.assert_array_equal(mean, self.processor.scaler_.mean_)

    def test_sorting_and_label_rules(self):
        base, x = self.processor.transform(self.frame)
        graphs = build_graphs(base, x)
        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0].x.shape[0], 3)
        self.assertEqual(graphs[0].y.ravel().tolist(), [1, 0, 1])
        ratio = build_graphs(base, x, label_rule='ratio', anomaly_ratio=0.5)
        self.assertEqual(ratio[0].y.ravel().tolist(), [0, 0, 1])
        edges = set(map(tuple, graphs[0].edge_index.t().tolist()))
        self.assertIn((0, 2), edges)
        self.assertIn((2, 0), edges)
        self.assertTrue(all((i, i) in edges for i in range(3)))
        engineered = self.processor._engineer(base)
        self.assertTrue((engineered.time_diff >= 0).all())

    def test_missing_columns_and_bad_labels_rejected(self):
        with self.assertRaises(ValueError):
            self.processor.transform(self.frame.drop(columns=['Bytes']))
        with self.assertRaises(ValueError):
            binary_labels(pd.DataFrame({'flag': [0, 2]}), 'flag')

    def test_iv_finite_with_pure_bins(self):
        self.assertTrue(np.isfinite(information_value(pd.Series([0, 0, 1, 1]), pd.Series([0, 0, 1, 1]))))

    def test_forward_is_repeatable_and_variance_is_trained(self):
        torch.manual_seed(42)
        graph = build_graphs(*self.processor.transform(self.frame))[0]
        model = STDNet(graph.x.shape[1], dict(hidden_dim=8, num_heads=2, d_ff=16, e_layers=1, dropout=0))
        original = graph.x.clone()
        model.eval()
        first = model(graph)
        second = model(graph)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(original, graph.x)
        self.assertEqual(first[0].shape, graph.y.shape)
        self.assertTrue((first[1] > 0).all())
        loss = HeteroscedasticFocalLoss()(first[0], first[1], graph.y)
        loss.backward()
        grad = model.scoring.var_layer.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_single_class_metrics(self):
        metrics = classification_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(metrics['roc_auc'])
        self.assertEqual(metrics['accuracy'], 1)

    def test_split_deduplicates_and_reuses_holdout(self):
        frame = pd.concat([self.frame] * 3, ignore_index=True)
        train, test = split_flows(frame, train_fraction=0.5, test_fraction=0.5)
        self.assertEqual(len(train) + len(test), len(self.frame))
        self.assertTrue(train.merge(test, how='inner').empty)
        larger = pd.concat([self.frame.assign(Bytes=self.frame.Bytes + i * 100) for i in range(10)])
        train70, test70 = split_flows(larger)
        train10, test10 = split_flows(larger, train_fraction=0.1)
        pd.testing.assert_frame_equal(test70, test10)
        self.assertLess(len(train10), len(train70))


if __name__ == '__main__':
    unittest.main()
