"""
Tests for the shipped model and the synthetic window dataset it was trained on
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import MLConfig
from src.core.anomaly_detector import MLEnhancedAnomalyDetector
from src.ml.feature_extractor import FeatureExtractor, FEATURE_SET_VERSION
from src.ml.ml_classifier import DDoSClassifier
from src.ml.traffic_dataset import (
    ALL_SCENARIOS, ATTACK_SCENARIOS, BENIGN_SCENARIOS,
    generate_dataset, generate_run, run_to_windows,
)

MODEL_PATH = Path(MLConfig.DEFAULT_MODEL_PATH)


@pytest.fixture(scope='module')
def classifier():
    if not MODEL_PATH.exists():
        pytest.skip('trained model not present')
    clf = DDoSClassifier()
    assert clf.load(str(MODEL_PATH))
    return clf


class TestTrafficDataset:
    def test_deterministic_for_seed(self):
        a = generate_dataset(5, seed=11)
        b = generate_dataset(5, seed=11)
        np.testing.assert_array_equal(a['X'], b['X'])
        np.testing.assert_array_equal(a['y'], b['y'])

    def test_labels_follow_scenarios(self):
        rng = np.random.default_rng(0)
        for scenario in ALL_SCENARIOS:
            run = generate_run(rng, scenario)
            X, y = run_to_windows(run)
            assert X.shape[1] == len(FeatureExtractor().feature_names)
            if scenario in BENIGN_SCENARIOS:
                assert y.sum() == 0, scenario
            else:
                assert y.sum() > 0, scenario
                assert (y == 0).sum() > 0, scenario  # pre-attack windows are benign

    def test_groups_identify_runs(self):
        data = generate_dataset(4, seed=3)
        assert set(data['groups']) == {0, 1, 2, 3}
        assert len(data['groups']) == len(data['y']) == len(data['scenarios'])


class TestShippedModel:
    def test_model_matches_runtime_features(self, classifier):
        assert FeatureExtractor().supports(classifier.feature_names)
        assert classifier.feature_set_version == FEATURE_SET_VERSION

    def test_model_has_held_out_evaluation(self, classifier):
        evaluation = classifier.evaluation
        assert evaluation and evaluation['data_source'] == 'synthetic-window'
        assert evaluation['test_metrics']['f1'] > 0.9
        assert classifier.metrics.accuracy == pytest.approx(evaluation['test_metrics']['accuracy'])

    def test_detector_loads_model(self):
        if not MODEL_PATH.exists():
            pytest.skip('trained model not present')
        detector = MLEnhancedAnomalyDetector(model_path=str(MODEL_PATH))
        assert detector.ml_enabled

    def test_unseen_runs_classified(self, classifier):
        """Fresh runs (seed not used in training) are classified well"""
        data = generate_dataset(40, seed=999)
        cols = [data['feature_names'].index(n) for n in classifier.feature_names]
        results = classifier.predict_batch(data['X'][:, cols])
        y_pred = np.array([r.is_attack for r in results])
        accuracy = (y_pred == data['y']).mean()
        assert accuracy > 0.95

    def test_predict_returns_python_types(self, classifier):
        x = np.zeros(len(classifier.feature_names))
        result = classifier.predict(x)
        assert type(result.is_attack) is bool
        assert 0.0 <= result.attack_probability <= 1.0

    def test_single_prediction_latency(self, classifier):
        x = np.zeros((1, len(classifier.feature_names)))
        classifier.predict(x)  # warmup
        latencies = [classifier.predict(x).inference_time_ms for _ in range(20)]
        assert np.median(latencies) < 20

    def test_old_cic_style_model_is_rejected(self, tmp_path):
        """A model trained on per-flow features must not be silently enabled"""
        clf = DDoSClassifier(n_estimators=5, max_depth=3)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        clf.train(X, y, feature_names=['Flow Duration', 'Total Fwd Packets', 'SYN Flag Count'])
        path = tmp_path / 'flow_model.joblib'
        clf.save(str(path))

        detector = MLEnhancedAnomalyDetector(model_path=str(path))
        assert not detector.ml_enabled
        assert detector.classifier is None
