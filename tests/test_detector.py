"""
Test Anomaly Detector - Unit tests for anomaly detection module
"""

import pytest
import time
from src.core.anomaly_detector import AnomalyDetector, MLEnhancedAnomalyDetector, TrafficBaseline, AnomalyScore


class TestAnomalyDetector:
    """Test cases for anomaly detector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = AnomalyDetector()
        assert detector.baseline.mean_pps == 0.0
        assert len(detector.history) == 0
    
    def test_baseline_update(self):
        """Test baseline updates"""
        detector = AnomalyDetector()
        
        # Simulate normal traffic
        for i in range(20):
            stats = {
                'total_packets': 10000 * (i + 1),
                'total_bytes': 1000000 * (i + 1),
                'tcp_packets': 8500,
                'udp_packets': 1000,
                'icmp_packets': 500,
            }
            detector.update_baseline(stats)
            time.sleep(0.01)  # Small delay
        
        # Baseline should be learned
        assert detector.baseline.samples >= 10
        assert detector.baseline.mean_pps > 0
    
    def test_high_packet_rate_detection(self):
        """Test detection of high packet rate"""
        detector = AnomalyDetector()
        
        # Establish baseline with low rate
        base_packets = 0
        for i in range(15):
            base_packets += 1000
            stats = {
                'total_packets': base_packets,
                'total_bytes': base_packets * 100,
                'tcp_packets': 850,
                'udp_packets': 100,
                'icmp_packets': 50,
            }
            detector.update_baseline(stats)
            time.sleep(0.01)
        
        # Inject high packet rate (attack)
        attack_stats = {
            'total_packets': base_packets + 1000000,  # Massive spike
            'total_bytes': (base_packets + 1000000) * 100,
            'tcp_packets': 850000,
            'udp_packets': 100000,
            'icmp_packets': 50000,
        }
        
        result = detector.detect_anomaly(attack_stats, [])
        
        assert result.is_anomaly == True
        assert result.score > 50
        assert any('high packet rate' in reason.lower() for reason in result.reasons)
    
    def test_protocol_anomaly_detection(self):
        """Test protocol distribution anomaly detection"""
        detector = AnomalyDetector()
        
        # Establish TCP-heavy baseline
        base_packets = 0
        for i in range(15):
            base_packets += 1000
            stats = {
                'total_packets': base_packets,
                'total_bytes': base_packets * 100,
                'tcp_packets': int(base_packets * 0.85),
                'udp_packets': int(base_packets * 0.10),
                'icmp_packets': int(base_packets * 0.05),
            }
            detector.update_baseline(stats)
            time.sleep(0.01)
        
        # UDP flood
        udp_flood_stats = {
            'total_packets': base_packets + 10000,
            'total_bytes': (base_packets + 10000) * 100,
            'tcp_packets': int(base_packets * 0.85) + 1000,
            'udp_packets': int(base_packets * 0.10) + 9000,  # 90% UDP now
            'icmp_packets': int(base_packets * 0.05),
        }
        
        result = detector.detect_anomaly(udp_flood_stats, [])
        
        assert any('udp' in reason.lower() for reason in result.reasons)
    
    def test_entropy_detection(self):
        """Test IP entropy detection"""
        detector = AnomalyDetector()
        
        # Low entropy (concentrated sources - botnet)
        low_entropy_ips = [
            {'ip': '192.168.1.1', 'packets': 10000},
            {'ip': '192.168.1.2', 'packets': 9000},
            {'ip': '192.168.1.3', 'packets': 8000},
        ]
        
        entropy = detector._calculate_ip_entropy(low_entropy_ips)
        assert entropy < 5.0  # Low entropy
        
        # High entropy (diverse sources - normal or flash crowd)
        high_entropy_ips = [
            {'ip': f'192.168.{i}.{j}', 'packets': 100}
            for i in range(10) for j in range(10)
        ]
        
        entropy = detector._calculate_ip_entropy(high_entropy_ips)
        assert entropy > 5.0  # Higher entropy
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism"""
        detector = AnomalyDetector()
        
        # First alert should go through
        assert detector.should_alert('test_alert') == True
        
        # Immediate second alert should be blocked
        assert detector.should_alert('test_alert') == False
        
        # Different alert type should go through
        assert detector.should_alert('other_alert') == True


def _cum(total, tcp=0.85, udp=0.10):
    return {
        'total_packets': total,
        'total_bytes': total * 100,
        'tcp_packets': int(total * tcp),
        'udp_packets': int(total * udp),
        'icmp_packets': total - int(total * tcp) - int(total * udp),
    }


class TestDetectorRegressions:
    """Behaviour fixed during the project review"""

    def test_first_sample_only_anchors(self):
        detector = AnomalyDetector()
        detector.update_baseline(_cum(10_000_000), now=50.0)
        assert len(detector.history) == 0
        result = detector.detect_anomaly(_cum(10_000_100), [], now=51.0)
        assert result.metrics['pps'] == pytest.approx(100)

    def test_protocol_ratio_uses_interval_not_lifetime(self):
        """A UDP flood after a long TCP-only history must show up immediately"""
        detector = AnomalyDetector()
        detector.update_baseline(_cum(1_000_000, tcp=1.0, udp=0.0), now=0.0)
        flood = _cum(1_000_000, tcp=1.0, udp=0.0)
        flood['total_packets'] += 5000
        flood['udp_packets'] += 5000
        result = detector.detect_anomaly(flood, [], now=1.0)
        assert result.metrics['udp_ratio'] == pytest.approx(1.0)
        assert any('udp' in r.lower() for r in result.reasons)

    def test_low_volume_skips_shape_heuristics(self):
        """A few pings from one host are not a 'concentrated ICMP flood'"""
        detector = AnomalyDetector()
        detector.update_baseline(_cum(0), now=0.0)
        stats = {'total_packets': 5, 'total_bytes': 320, 'tcp_packets': 0,
                 'udp_packets': 0, 'icmp_packets': 5}
        result = detector.detect_anomaly(stats, [{'ip': '10.0.0.5', 'packets': 5}], now=1.0)
        assert result.score == 0
        assert not result.is_anomaly

    def test_traffic_drop_is_not_an_attack(self):
        detector = AnomalyDetector()
        t, total = 0.0, 0
        for _ in range(15):
            total += 1000
            detector.update_baseline(_cum(total), now=t)
            t += 1
        result = detector.detect_anomaly(_cum(total + 10), [], now=t)
        assert 'pps_sigma' in result.metrics and result.metrics['pps_sigma'] < 0
        assert not any('deviation' in r for r in result.reasons)

    def test_idle_second_before_flood_keeps_change_rule(self):
        """Regression (live XDP test): an idle second just before a flood switched
        the rate-of-change rule off, so a 1,900 pps UDP flood scored 85 (< 90)"""
        detector = AnomalyDetector()
        t, total = 0.0, 0
        for _ in range(15):
            total += 5
            detector.update_baseline(_cum(total, tcp=0.0, udp=0.0), now=t)
            t += 1
        detector.update_baseline(_cum(total, tcp=0.0, udp=0.0), now=t)  # idle second
        t += 1
        flood = _cum(total, tcp=0.0, udp=0.0)
        flood['total_packets'] += 1904
        flood['udp_packets'] += 1904
        result = detector.detect_anomaly(flood, [], now=t)
        assert any('Rapid traffic increase' in r for r in result.reasons)
        assert result.score >= MLEnhancedAnomalyDetector.STAT_ONLY_SCORE

    def test_learn_false_keeps_baseline(self):
        detector = AnomalyDetector()
        t, total = 0.0, 0
        for _ in range(15):
            total += 100
            detector.update_baseline(_cum(total), now=t)
            t += 1
        mean_before = detector.baseline.mean_pps
        detector.update_baseline(_cum(total + 1_000_000), now=t, learn=False)
        assert detector.baseline.mean_pps == mean_before
        assert detector.last_stats['total_packets'] == total + 1_000_000


class TestHybridCombination:
    """MLEnhancedAnomalyDetector._combine_results decision rules"""

    @staticmethod
    def _ml(is_attack, confidence):
        from src.ml.ml_classifier import PredictionResult
        return PredictionResult(is_attack=is_attack, attack_probability=confidence / 100 if is_attack else 1 - confidence / 100,
                                attack_type='UDP_Flood' if is_attack else 'BENIGN',
                                confidence=confidence, inference_time_ms=0.1)

    @staticmethod
    def _stat(score):
        return AnomalyScore(is_anomaly=score >= 50, score=score, reasons=[], metrics={})

    def test_ml_vetoes_moderate_statistical_detection(self):
        from src.core.anomaly_detector import MLEnhancedAnomalyDetector
        detector = MLEnhancedAnomalyDetector()
        result = detector._combine_results(self._stat(65), self._ml(False, 95))
        assert not result.is_anomaly

    def test_strong_statistical_detection_stands(self):
        from src.core.anomaly_detector import MLEnhancedAnomalyDetector
        detector = MLEnhancedAnomalyDetector()
        result = detector._combine_results(self._stat(100), self._ml(False, 95))
        assert result.is_anomaly and result.detection_source == 'statistical'
        assert result.attack_type == 'UNKNOWN'

    def test_confident_ml_detection_alone(self):
        from src.core.anomaly_detector import MLEnhancedAnomalyDetector
        detector = MLEnhancedAnomalyDetector()
        result = detector._combine_results(self._stat(0), self._ml(True, 90))
        assert result.is_anomaly and result.detection_source == 'ml'

    def test_agreement_is_hybrid(self):
        from src.core.anomaly_detector import MLEnhancedAnomalyDetector
        detector = MLEnhancedAnomalyDetector()
        result = detector._combine_results(self._stat(55), self._ml(True, 60))
        assert result.is_anomaly and result.detection_source == 'hybrid'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
