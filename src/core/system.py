"""
DDoS Mitigation System - orchestrator

Wires the components together and runs one monitoring interval at a time:
TrafficMonitor (eBPF counters) -> AnomalyDetector (statistical + ML)
-> mitigation -> AlertSystem / TrafficProfiler.

Mitigation policies:
  p4    source scoring (default): blocks confident attackers for 30 s, slows
        uncertain sources, and limits brand-new sources during spoofed floods
  top5  the original rule: permanently block the 5 busiest sources per alert

Model sets (--models):
  synthetic  (default) detector, scorer and thresholds trained on synthetic traffic
  cic        with real captures (CIC-DDoS2019 day 01-12, scripts/cic_experiment.py):
             window detector trained on synthetic + CIC traffic, the level-3 flow
             classifier trained on CIC flows verifying P4 decisions (P4+L3, needs live
             XDP flow counters), and the synthetic-trained scorer and thresholds (a
             scorer trained on CIC learns its single attack machine and does not
             transfer; data/results/cic_results.json, level2.transfer_to_synthetic)
"""

import logging
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import coloredlogs

from src.config import (
    DATA_DIR, EBPF_PROGRAM_PATH, MLConfig, MODELS_DIR, MonitoringConfig, PLATFORM,
    DetectionThresholds, TimeWindows,
)
from src.core.traffic_monitor import TrafficMonitor, BCC_AVAILABLE
from src.core.anomaly_detector import AnomalyDetector, MLEnhancedAnomalyDetector, AnomalyScore
from src.core.traffic_profiler import TrafficProfiler
from src.core.alert_system import AlertSystem
from src.core.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

POLICIES = {'p4': 'P4 source scoring', 'top5': 'Top-5 blacklist (permanent)'}
SCORER_PATH = MODELS_DIR / 'source_scorer.joblib'
RESULTS_PATH = DATA_DIR / 'results' / 'mitigation_results.json'
CIC_RESULTS_PATH = DATA_DIR / 'results' / 'cic_results.json'
MODEL_SETS = {
    'synthetic': {'window': Path(MLConfig.DEFAULT_MODEL_PATH), 'scorer': SCORER_PATH, 'flow': None},
    'cic': {'window': MODELS_DIR / 'cic_window_model.joblib', 'scorer': SCORER_PATH,
            'flow': MODELS_DIR / 'cic_flow_model.joblib'},
}


def cic_choices(system: str = 'cic_set'):
    """P4 thresholds and the P4+L3 mode chosen on CIC day 01-12 by scripts/cic_experiment.py
    for an evaluated system ('cic_set' or 'shipped'); (None, 'limit') if it has not been run"""
    import json
    try:
        res = json.loads(CIC_RESULTS_PATH.read_text())['level2']['systems'][system]
        return ({k: float(v) for k, v in res['thresholds'].items()},
                res.get('p4_l3_on_legit_flows', 'limit'))
    except (OSError, KeyError, ValueError, TypeError):
        return None, 'limit'


def setup_logging():
    """UTF-8 log file plus coloured console output (one handler each)"""
    Path(MonitoringConfig.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, MonitoringConfig.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(MonitoringConfig.LOG_FILE, encoding='utf-8')],
        force=True,
    )
    coloredlogs.install(
        level=MonitoringConfig.LOG_LEVEL,
        fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'
    )


class DDoSMitigationSystem:
    """Main application class - Phase 2 with ML support"""
    
    def __init__(self, interface: str, xdp_mode: str = 'native', ml_model_path: str = None,
                 allow_simulation_api: bool = False, auto_detect_model: bool = True,
                 policy: str = 'p4', models: str = 'synthetic', flow_check: bool = None):
        if models not in MODEL_SETS:
            raise ValueError(f"unknown model set {models!r}; choose from: " + ', '.join(MODEL_SETS))
        self.models = models
        self.interface = interface
        self.xdp_mode = xdp_mode
        self.running = False
        self.ml_enabled = False
        self.ml_model_path = ml_model_path
        # Injecting fake stats can blacklist arbitrary IPs, so on a live XDP
        # deployment the simulation API must be enabled explicitly
        self.allow_simulation_api = allow_simulation_api

        # Detection state is shared by the monitoring loop and the dashboard thread
        self._lock = threading.RLock()
        self._last_sample_time = None
        self._current_pps = 0.0
        self._current_bps = 0.0
        self._current_drop_pps = 0.0
        self._last_top_ips = []
        self.last_result = None
        # For the dashboard: one point per interval (chart) and a short event log
        self._timeline = deque(maxlen=180)
        self._events = deque(maxlen=60)
        self._in_attack = False
        self.demo = None   # DemoRunner, attached by the dashboard in simulation mode
        self.p4 = None
        # Level 3: flow classifier verifying P4 (on by default with the cic model set)
        self.flow_evidence = None
        if flow_check if flow_check is not None else models == 'cic':
            from src.mitigation.live import LiveFlowEvidence
            self.flow_evidence = LiveFlowEvidence.load(MODEL_SETS['cic']['flow'])
        self.set_policy(policy)

        # Initialize components
        self.traffic_monitor = None
        
        # Auto-detect ML model if not provided
        if not ml_model_path and auto_detect_model:
            default_model = MODEL_SETS[models]['window']
            if default_model.exists():
                ml_model_path = str(default_model)
                logger.info(f"Using auto-detected ML model: {ml_model_path}")
        
        # Use ML-enhanced detector if model provided
        if ml_model_path:
            self.anomaly_detector = MLEnhancedAnomalyDetector(model_path=ml_model_path)
            self.ml_enabled = self.anomaly_detector.ml_enabled
        else:
            self.anomaly_detector = AnomalyDetector()
        
        self.traffic_profiler = TrafficProfiler()
        self.alert_system = AlertSystem()
        self.metrics_collector = MetricsCollector()
    
    def set_policy(self, policy: str) -> str:
        """Choose the mitigation policy ('p4' or 'top5'); falls back to top5 without a scorer"""
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choose from: " + ', '.join(POLICIES))
        with self._lock:
            self.p4 = None
            if policy == 'p4':
                from src.mitigation.live import LiveSourceScoring
                scorer_path = MODEL_SETS[self.models]['scorer']
                # thresholds always from the synthetic tuning (RESULTS_PATH); the P4+L3 mode was
                # chosen on CIC 01-12 for this system
                _, legit_mode = cic_choices('cic_set' if self.models == 'cic' else 'shipped')
                self.p4 = LiveSourceScoring.load(scorer_path, RESULTS_PATH,
                                                 flow_verified=self.flow_evidence is not None,
                                                 on_legit_flows=legit_mode)
                if self.p4 is None:
                    logger.warning(f"P4 needs {scorer_path} (python scripts/mitigation_experiment.py); "
                                   "using top5")
                    policy = 'top5'
            self.mitigation_policy = policy
        logger.info(f"Mitigation policy: {POLICIES[policy]}")
        return policy

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, stopping...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the DDoS mitigation system"""
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("=" * 70)
        if self.ml_enabled:
            logger.info("DDoS Mitigation System - Phase 2: ML-Based Classification")
        else:
            logger.info("DDoS Mitigation System - Phase 1: Baseline Anomaly Detection")
        logger.info("=" * 70)
        logger.info(f"Platform: {PLATFORM}")
        logger.info(f"Interface: {self.interface}")
        logger.info(f"XDP Mode: {self.xdp_mode}")
        logger.info(f"ML Enabled: {self.ml_enabled}")
        if self.ml_enabled and hasattr(self.anomaly_detector, 'classifier'):
            if self.anomaly_detector.classifier and self.anomaly_detector.classifier.metrics:
                logger.info(f"ML Model Accuracy: {self.anomaly_detector.classifier.metrics.accuracy:.4f}")
        logger.info("=" * 70)
        
        # Check for eBPF support
        if not BCC_AVAILABLE:
            if PLATFORM == 'windows':
                logger.warning("BCC/eBPF not natively available on Windows - Using Simulation Mode")
            else:
                logger.error("BCC/eBPF not available!")
                logger.error("Please install: sudo apt-get install python3-bpfcc (Linux)")
                logger.error("Or: Install Microsoft eBPF runtime (Windows)")
                return False
        
        # Initialize traffic monitor
        try:
            self.traffic_monitor = TrafficMonitor(self.interface, self.xdp_mode)
            
            # Check if eBPF program exists
            if not Path(EBPF_PROGRAM_PATH).exists():
                logger.error(f"eBPF program not found: {EBPF_PROGRAM_PATH}")
                logger.error("Please compile: cd src/ebpf && make")
                return False
            
            # Load XDP program
            if not self.traffic_monitor.load_xdp_program(EBPF_PROGRAM_PATH):
                logger.error("Failed to load XDP program")
                return False
            
            if self.traffic_monitor.simulation_mode:
                logger.info("Simulation mode active (no XDP program attached)")
                self.allow_simulation_api = True
            else:
                logger.info("XDP program loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize traffic monitor: {e}")
            return False
        
        # Start monitoring loop
        self.running = True
        self._monitoring_loop()
        
        return True
    
    def process_sample(self, stats: dict, ip_stats: list, label: str = None,
                       now: float = None) -> AnomalyScore:
        """
        Run one monitoring interval through detection, mitigation and alerting.

        Args:
            stats: Cumulative counters (total_packets, total_bytes, tcp_packets, ...)
            ip_stats: Per-source-IP counts for THIS interval
            label: Optional tag for simulated samples (shown in alerts)
            now: Sample timestamp (defaults to time.time())
        """
        now = time.time() if now is None else now
        with self._lock:
            if self.traffic_monitor is not None:
                self.traffic_monitor.sim_now = now   # expiry clock of simulated kernel maps
            prev_time = self._last_sample_time
            prev_stats = self.anomaly_detector.last_stats or {}
            if prev_time is not None and now > prev_time:
                elapsed = now - prev_time
                self._current_pps = max(0, stats.get('total_packets', 0) - prev_stats.get('total_packets', 0)) / elapsed
                self._current_bps = max(0, stats.get('total_bytes', 0) - prev_stats.get('total_bytes', 0)) / elapsed
                self._current_drop_pps = max(0, stats.get('dropped_packets', 0) - prev_stats.get('dropped_packets', 0)) / elapsed
            self._last_sample_time = now

            # Detect on the new interval first, then advance the baseline;
            # anomalous intervals are not learned so attacks can't poison it
            result = self.anomaly_detector.detect_anomaly(stats, ip_stats, now=now)
            self.anomaly_detector.update_baseline(stats, now=now, learn=not result.is_anomaly)
            self.traffic_profiler.update_profile(stats, self._current_pps, self._current_bps)
            self.last_result = result
            top_ips = sorted(ip_stats, key=lambda x: x.get('packets', 0), reverse=True)
            self._last_top_ips = top_ips[:10]

            if prev_time is not None:   # the first sample only anchors the counters
                self._record_interval(now, result, label)

            if self.p4 is not None:
                # Level 3: classify the flows of the last 10-s window (live XDP only)
                if self.flow_evidence is not None and self.traffic_monitor is not None:
                    self.flow_evidence.maybe_update(now, self.traffic_monitor)
                # P4 sees every interval: normal ones build each source's history
                outcome = self.p4.step(now, ip_stats, result.is_anomaly, self.traffic_monitor,
                                       flows=self.flow_evidence)
                self._note_mitigation(now, len(outcome['blocked']), len(outcome['limited']))

            if result.is_anomaly and self.p4 is None:
                # Blacklist the 5 busiest sources not blocked yet, even when alerts are
                # in cooldown. XDP counts packets before dropping them, so blocked
                # sources stay at the top of the list and must be skipped, not counted.
                newly_blocked = 0
                for ip_stat in top_ips:
                    if newly_blocked >= 5:
                        break
                    ip = ip_stat.get('ip')
                    if ip_stat.get('packets', 0) <= DetectionThresholds.BLACKLIST_MIN_PACKETS:
                        break  # sorted by packets, so the rest are smaller
                    if self.traffic_monitor.is_blacklisted(ip):
                        continue
                    if self.traffic_monitor.add_to_blacklist(ip):
                        newly_blocked += 1
                        logger.warning(f"Mitigation: Blacklisted {ip} "
                                       f"({ip_stat['packets']} packets this interval)")
                self._note_mitigation(now, newly_blocked, 0)

            if result.is_anomaly:
                if self.anomaly_detector.should_alert('ddos_attack'):
                    prefix = f"[SIM] {label}: " if label else ""
                    self.alert_system.send_alert(
                        alert_type='ddos_attack',
                        severity='high' if result.score >= 75 else 'medium',
                        message=f"{prefix}DDoS attack detected (score: {result.score:.1f}, "
                                f"source: {result.detection_source}, type: {result.attack_type})",
                        details={
                            'reasons': result.reasons,
                            'metrics': result.metrics,
                            'current_pps': self._current_pps,
                            'baseline_pps': self.anomaly_detector.baseline.mean_pps,
                        }
                    )
            return result

    def _note_mitigation(self, now: float, blocked: int, limited: int):
        """One running event-log entry per attack instead of one line per second"""
        if not blocked and not limited:
            return
        last_event = self._events[-1] if self._events else None
        if last_event and last_event['kind'] == 'block':
            last_event['count'] += blocked
            last_event['limited'] = last_event.get('limited', 0) + limited
            last_event['t'] = now
        else:
            self._events.append({'t': now, 'kind': 'block', 'count': blocked, 'limited': limited})
            last_event = self._events[-1]
        n, m = last_event['count'], last_event['limited']
        parts = []
        if n:
            ttl = f" for {self.p4.ttl:g} s" if self.p4 is not None else ""
            parts.append(f"Blocked {n} source{'s' if n > 1 else ''}{ttl}")
        if m:
            parts.append(f"slowed {m} source{'s' if m > 1 else ''}")
        last_event['text'] = ', '.join(parts) + ' in the kernel (XDP)'

    def _add_event(self, t: float, kind: str, text: str):
        self._events.append({'t': t, 'kind': kind, 'text': text})

    def _record_interval(self, now: float, result: AnomalyScore, label: str = None):
        """Chart point for this interval, plus start/end-of-attack events"""
        self._timeline.append({
            't': now,
            'pps': round(self._current_pps, 1),
            'dropped_pps': round(self._current_drop_pps, 1),
            'baseline_pps': round(self.anomaly_detector.baseline.mean_pps, 1),
            'anomaly': bool(result.is_anomaly),
            'score': round(result.score, 1),
        })
        if result.is_anomaly and not self._in_attack:
            kind = result.attack_type if result.attack_type not in ('UNKNOWN', 'BENIGN') else 'DDoS'
            decided_by = {'statistical': 'rules', 'ml': 'ML model', 'hybrid': 'rules + ML'}.get(
                result.detection_source, result.detection_source)
            self._add_event(now, 'attack', f"Attack detected: {kind.replace('_', ' ')} at "
                            f"{self._current_pps:,.0f} packets/s (decided by {decided_by})")
        elif self._in_attack and not result.is_anomaly:
            self._add_event(now, 'clear', 'Traffic back to normal')
        self._in_attack = bool(result.is_anomaly)

    def reset_demo_state(self):
        """Clear blocks, alerts, chart and events (simulation mode only)"""
        with self._lock:
            monitor = self.traffic_monitor
            if monitor and monitor.simulation_mode:
                for ip in monitor.get_blacklist():
                    monitor.remove_from_blacklist(ip)
                monitor._sim_blacklist.clear()
                monitor._sim_limits.clear()
                monitor._sim_known.clear()
                monitor._sim_guard = None
            if self.p4 is not None:
                self.p4._reset_index()
            self.alert_system.alert_history.clear()
            self.anomaly_detector.alert_cooldown.clear()
            self._timeline.clear()
            self._events.clear()
            self._in_attack = False
            self.last_result = None

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring loop...")
        if self.traffic_monitor.simulation_mode:
            logger.info("Simulation mode: waiting for samples via POST /api/simulate_attack "
                        "(e.g. python run_attack_demo.py)")

        iteration = 0

        while self.running:
            try:
                iteration += 1

                # Live XDP: read kernel counters and process the interval.
                # Simulation mode has no kernel data; samples arrive via the API.
                if not self.traffic_monitor.simulation_mode:
                    stats = self.traffic_monitor.get_statistics()
                    ip_stats = self.traffic_monitor.get_ip_interval_statistics(limit=1000)
                    if stats:
                        self.process_sample(stats, ip_stats)

                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()

                # Log status (every 10 iterations)
                if iteration % 10 == 0:
                    stats = self.traffic_monitor.get_statistics()
                    logger.info(f"Status: {self._current_pps:.0f} pps | "
                               f"Baseline: {self.anomaly_detector.baseline.mean_pps:.0f} pps | "
                               f"Blacklisted: {len(self.traffic_monitor.get_blacklist())} | "
                               f"Dropped: {stats.get('dropped_packets', 0)} | "
                               f"CPU: {system_metrics.cpu_percent:.1f}%")
                
                # Sleep until next update
                time.sleep(TimeWindows.STATISTICS_UPDATE)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(1)
        
        logger.info("Monitoring loop stopped")
    
    def stop(self):
        """Stop the system and cleanup"""
        logger.info("Stopping DDoS mitigation system...")
        self.running = False
        
        if self.traffic_monitor:
            self.traffic_monitor.unload_xdp_program()
        
        # Save profile
        if self.traffic_profiler:
            self.traffic_profiler.save_profile()
        
        logger.info("System stopped")
    
    def get_status(self) -> dict:
        """Get current system status"""
        with self._lock:
            monitor = self.traffic_monitor
            simulation = bool(monitor and monitor.simulation_mode)
            stats = monitor.get_statistics() if monitor else {}
            if simulation:
                # No kernel counters: show the last injected cumulative sample
                stats = dict(self.anomaly_detector.last_stats or {})
            profile = self.traffic_profiler.get_profile()
            last = self.last_result
            blacklist = monitor.get_blacklist() if monitor else []
            blocked = set(blacklist)
            limited = set(monitor.get_rate_limited()) if monitor else set()
            top_ips = [dict(entry, blocked=entry.get('ip') in blocked,
                            limited=entry.get('ip') in limited and entry.get('ip') not in blocked)
                       for entry in self._last_top_ips]

            status = {
                'running': self.running,
                'interface': self.interface,
                'simulation_mode': simulation,
                'statistics': stats,
                'baseline': self.anomaly_detector.get_baseline_info(),
                'profile': profile.__dict__ if profile else {},
                'blacklist': blacklist,
                'mitigation': {
                    'policy': self.mitigation_policy,
                    'title': POLICIES[self.mitigation_policy],
                    'blocked': len(blacklist),
                    'limited': len(limited - blocked),
                    'spoof_guard': bool(monitor and monitor.spoof_guard_active()),
                    'models': self.models,
                    'flow_check': self.flow_evidence is not None,
                    'flows_classified': self.flow_evidence.flows_classified if self.flow_evidence else 0,
                },
                'recent_alerts': self.alert_system.get_recent_alerts(10),
                'ip_stats': top_ips,  # top sources in the latest interval
                'current_pps': self._current_pps,
                'current_bps': self._current_bps,
                'current_drop_pps': self._current_drop_pps,
                'timeline': list(self._timeline),
                'events': list(self._events)[::-1],
                'demo': self.demo.status() if self.demo else None,
                'allow_simulation_api': self.allow_simulation_api,
                'last_detection': {
                    'is_anomaly': last.is_anomaly,
                    'score': last.score,
                    'source': last.detection_source,
                    'attack_type': last.attack_type,
                    'reasons': last.reasons,
                    'ml_prediction': last.ml_prediction,
                    'ml_confidence': last.ml_confidence,
                    'pps': last.metrics.get('pps'),
                } if last else None,
                'ml_enabled': self.ml_enabled,
                'ml_stats': {},
                'feature_importance': {},
            }

            if hasattr(self.anomaly_detector, 'get_ml_stats'):
                status['ml_stats'] = self.anomaly_detector.get_ml_stats()
                status['feature_importance'] = self.anomaly_detector.get_feature_importance(10)

        return status
