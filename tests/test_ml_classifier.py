"""
Test ML Classifier - Unit tests for Phase 2 ML classification module
"""

import pytest
import numpy as np
import time
import tempfile
import os
from pathlib import Path

# Import ML modules
from src.config import MLConfig
from src.ml.data_loader import CICDataLoader, create_synthetic_dataset, SELECTED_FEATURES
from src.ml.feature_extractor import FeatureExtractor, FlowFeatures
from src.ml.ml_classifier import DDoSClassifier, PredictionResult, ModelMetrics


class TestCICDataLoader:
    """Tests for CIC-DDoS-2019 data loader"""
    
    def test_initialization(self):
        """Test loader initialization"""
        loader = CICDataLoader()
        assert loader.data_dir == Path(MLConfig.DATASET_DIR)
        assert len(loader.feature_names) > 0
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset generation"""
        df = create_synthetic_dataset(1000)
        
        assert len(df) == 1000
        assert ' Label' in df.columns
        
        # Check for both benign and attack samples
        labels = df[' Label'].unique()
        assert 'BENIGN' in labels
    
    def test_preprocessing(self):
        """Test data preprocessing"""
        loader = CICDataLoader()
        df = create_synthetic_dataset(500)
        
        X, y_str, feature_names = loader.preprocess(df)
        
        assert len(X) > 0
        assert len(y_str) == len(X)
        assert len(feature_names) > 0
        assert X.dtype == np.float32
    
    def test_label_encoding_binary(self):
        """Test binary label encoding"""
        loader = CICDataLoader()
        labels = np.array(['BENIGN', 'DrDoS_UDP', 'BENIGN', 'Syn'])
        
        encoded = loader.encode_labels(labels, binary=True)
        
        assert len(encoded) == 4
        assert encoded[0] == 0  # BENIGN
        assert encoded[1] == 1  # Attack
        assert encoded[2] == 0  # BENIGN
        assert encoded[3] == 1  # Attack
    
    def test_feature_scaling(self):
        """Test feature scaling"""
        loader = CICDataLoader()
        
        X_train = np.random.randn(100, 10).astype(np.float32)
        X_test = np.random.randn(20, 10).astype(np.float32)
        
        X_train_scaled, X_test_scaled = loader.scale_features(X_train, X_test, fit=True)
        
        # Scaled data should have mean close to 0 and std close to 1
        assert np.abs(X_train_scaled.mean()) < 0.5
        assert X_test_scaled is not None


class TestFeatureExtractor:
    """Tests for real-time feature extraction"""
    
    def test_initialization(self):
        """Test extractor initialization"""
        extractor = FeatureExtractor(window_size=10.0)
        
        assert extractor.window_size == 10.0
        assert len(extractor.feature_names) > 0
    
    @staticmethod
    def _feed(extractor, seconds, pps=1000, tcp=0.85, udp=0.10, n_ips=10, syn_per_ip=2, t0=1000.0):
        """Feed `seconds` one-second intervals of steady traffic"""
        for i in range(seconds + 1):
            n = pps * i
            stats = {
                'total_packets': n,
                'total_bytes': n * 100,
                'tcp_packets': int(n * tcp),
                'udp_packets': int(n * udp),
                'icmp_packets': n - int(n * tcp) - int(n * udp),
            }
            ip_stats = [{'ip': f'10.0.0.{j}', 'packets': pps // n_ips, 'syn_count': syn_per_ip}
                        for j in range(n_ips)]
            extractor.update(stats, ip_stats, timestamp=t0 + i)
        return t0 + seconds

    def test_update_with_stats(self):
        """First sample anchors counters; the second creates an interval"""
        extractor = FeatureExtractor()
        stats = {'total_packets': 1000, 'total_bytes': 100000, 'tcp_packets': 850,
                 'udp_packets': 100, 'icmp_packets': 50}

        extractor.update(stats, timestamp=100.0)
        assert len(extractor.window) == 0

        stats2 = {k: v * 2 for k, v in stats.items()}
        extractor.update(stats2, timestamp=101.0)
        assert len(extractor.window) == 1
        assert extractor.total_packets == 1000

    def test_extract_features(self):
        """Window features reflect the traffic that was fed in"""
        extractor = FeatureExtractor()
        end = self._feed(extractor, 5, pps=1000, n_ips=10)

        features = extractor.extract_features(end)
        values = features.to_dict()

        assert isinstance(features, FlowFeatures)
        assert len(features.features) == len(extractor.feature_names)
        assert values['pps_mean'] == pytest.approx(1000)
        assert values['tcp_ratio'] == pytest.approx(0.85, abs=0.01)
        assert values['unique_src_ips'] == pytest.approx(10)
        assert values['src_ip_entropy_norm'] == pytest.approx(1.0)  # equal shares
        assert values['top1_share'] == pytest.approx(0.1)
        assert values['syn_share'] == pytest.approx(2 / 100)

    def test_window_expires_old_samples(self):
        """Only samples inside window_size seconds are used"""
        extractor = FeatureExtractor(window_size=5.0)
        self._feed(extractor, 20)
        assert len(extractor.window) == 5

    def test_counter_reset_is_not_negative(self):
        """A counter that goes backwards is treated as a reset, not a negative rate"""
        extractor = FeatureExtractor()
        extractor.update({'total_packets': 5000, 'total_bytes': 1}, timestamp=1.0)
        extractor.update({'total_packets': 100, 'total_bytes': 1}, timestamp=2.0)
        assert extractor.extract_features(2.0).to_dict()['pps_mean'] == pytest.approx(100)

    def test_features_for_prediction(self):
        """Test preparing features for ML prediction"""
        extractor = FeatureExtractor()
        end = self._feed(extractor, 3)

        features = extractor.extract_features_for_prediction(timestamp=end)

        assert features.shape[0] == 1  # Batch size 1
        assert features.shape[1] == len(extractor.feature_names)

    def test_features_for_prediction_selects_model_columns(self):
        """Columns follow the model's feature order, and unknown features fail loudly"""
        extractor = FeatureExtractor()
        end = self._feed(extractor, 3)

        row = extractor.extract_features_for_prediction(feature_names=['udp_ratio', 'pps_mean'],
                                                        timestamp=end)
        assert row.shape == (1, 2)
        assert row[0, 1] == pytest.approx(1000)

        with pytest.raises(ValueError):
            extractor.extract_features_for_prediction(feature_names=['Flow Duration'])
        assert not extractor.supports(['Flow Duration'])

    def test_reset(self):
        """Test extractor reset"""
        extractor = FeatureExtractor()
        self._feed(extractor, 3)

        extractor.reset()

        assert len(extractor.window) == 0
        assert extractor.total_packets == 0


class TestDDoSClassifier:
    """Tests for ML classifier"""
    
    def test_initialization(self):
        """Test classifier initialization"""
        classifier = DDoSClassifier(
            model_type='random_forest',
            n_estimators=10,
            max_depth=5,
        )
        
        assert classifier.model_type == 'random_forest'
        assert classifier.n_estimators == 10
        assert classifier.is_trained == False
    
    def test_training(self):
        """Test model training"""
        classifier = DDoSClassifier(n_estimators=10, max_depth=5)
        
        # Create synthetic training data
        n_samples = 200
        n_features = 20
        
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, n_samples)
        
        X_val = np.random.randn(50, n_features).astype(np.float32)
        y_val = np.random.randint(0, 2, 50)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        metrics = classifier.train(X_train, y_train, X_val, y_val, feature_names)
        
        assert classifier.is_trained == True
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert len(classifier.feature_names) == n_features
    
    def test_prediction(self):
        """Test single sample prediction"""
        classifier = DDoSClassifier(n_estimators=10, max_depth=5)
        
        # Train first
        n_features = 20
        X_train = np.random.randn(100, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        classifier.train(X_train, y_train)
        
        # Predict
        X_test = np.random.randn(n_features).astype(np.float32)
        result = classifier.predict(X_test)
        
        assert isinstance(result, PredictionResult)
        assert isinstance(result.is_attack, bool)
        assert 0 <= result.confidence <= 100
        assert result.inference_time_ms >= 0
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        classifier = DDoSClassifier(n_estimators=10, max_depth=5)
        
        n_features = 20
        X_train = np.random.randn(100, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        classifier.train(X_train, y_train)
        
        # Batch predict
        X_test = np.random.randn(10, n_features).astype(np.float32)
        results = classifier.predict_batch(X_test)
        
        assert len(results) == 10
        assert all(isinstance(r, PredictionResult) for r in results)
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        classifier = DDoSClassifier(n_estimators=10, max_depth=5)
        
        n_features = 20
        X_train = np.random.randn(100, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        classifier.train(X_train, y_train, feature_names=feature_names)
        
        importance = classifier.get_feature_importance(top_n=5)
        
        assert len(importance) == 5
        assert all(0 <= v <= 1 for v in importance.values())
    
    def test_save_and_load(self):
        """Test model persistence"""
        classifier = DDoSClassifier(n_estimators=10, max_depth=5)
        
        # Train
        n_features = 20
        X_train = np.random.randn(100, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        classifier.train(X_train, y_train)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            assert classifier.save(model_path) == True
            
            # Load in new classifier
            new_classifier = DDoSClassifier()
            assert new_classifier.load(model_path) == True
            assert new_classifier.is_trained == True
            
            # Verify predictions match
            X_test = np.random.randn(1, n_features).astype(np.float32)
            result1 = classifier.predict(X_test)
            result2 = new_classifier.predict(X_test)
            
            assert result1.is_attack == result2.is_attack
    
    def test_untrained_prediction(self):
        """Test prediction on untrained model returns default"""
        classifier = DDoSClassifier()
        
        X_test = np.random.randn(20).astype(np.float32)
        result = classifier.predict(X_test)
        
        assert result.is_attack == False
        assert result.confidence == 0.0
        assert result.attack_type == 'UNKNOWN'


class TestMLIntegration:
    """Integration tests for ML module"""
    
    def test_end_to_end_pipeline(self):
        """Test complete ML pipeline"""
        # 1. Create synthetic dataset
        df = create_synthetic_dataset(500)
        
        # 2. Load and preprocess
        loader = CICDataLoader()
        X, y_str, feature_names = loader.preprocess(df)
        y = loader.encode_labels(y_str, binary=True)
        
        # 3. Scale features
        X_scaled, _ = loader.scale_features(X, fit=True)
        
        # 4. Train classifier
        classifier = DDoSClassifier(n_estimators=20, max_depth=10)
        classifier.scaler = loader.scaler
        
        # Split data
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        metrics = classifier.train(X_train, y_train, X_test, y_test, feature_names)
        
        # 5. Make predictions
        result = classifier.predict(X_test[0])
        
        assert classifier.is_trained
        assert metrics.accuracy > 0.5  # Should be better than random
        assert isinstance(result.is_attack, bool)
    
    def test_feature_extractor_to_classifier(self):
        """Test feature extraction to classification pipeline"""
        from src.ml.feature_extractor import FeatureExtractor
        
        # Simulate traffic
        extractor = FeatureExtractor()
        for i in range(10):
            stats = {
                'total_packets': 1000 * i,
                'total_bytes': 100000 * i,
                'tcp_packets': 800 * i,
            }
            extractor.update(stats, [{'ip': '10.0.0.1', 'packets': 1000, 'syn_count': 50}],
                             timestamp=100.0 + i)
        
        # Train a small classifier
        n_features = len(extractor.feature_names)
        X_train = np.random.randn(100, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        
        classifier = DDoSClassifier(n_estimators=10, max_depth=5)
        classifier.train(X_train, y_train, feature_names=extractor.feature_names)
        
        # Extract features and predict
        features = extractor.extract_features_for_prediction()
        result = classifier.predict(features)
        
        assert isinstance(result, PredictionResult)


class TestInferenceLatency:
    """Performance tests for inference latency"""
    
    def test_single_prediction_latency(self):
        """Test single prediction is fast enough"""
        classifier = DDoSClassifier(n_estimators=50, max_depth=10)
        
        n_features = 64  # CIC-DDoS-2019 feature count
        X_train = np.random.randn(1000, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, 1000)
        classifier.train(X_train, y_train)
        
        # Warmup
        X_test = np.random.randn(n_features).astype(np.float32)
        classifier.predict(X_test)
        
        # Measure
        latencies = []
        for _ in range(100):
            X_test = np.random.randn(n_features).astype(np.float32)
            result = classifier.predict(X_test)
            latencies.append(result.inference_time_ms)
        
        avg_latency = np.mean(latencies)
        
        # Should be fast enough for real-time (< 10ms)
        assert avg_latency < 10, f"Inference too slow: {avg_latency:.2f}ms"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
