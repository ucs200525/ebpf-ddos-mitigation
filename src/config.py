"""
Configuration file for DDoS Mitigation System
"""

import os
import platform
from pathlib import Path

# Project layout: every path below is absolute, derived from this file's
# location, so the system works regardless of the current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../project
SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = DATA_DIR / 'models'
LOG_DIR = PROJECT_ROOT / 'logs'
BLACKLIST_FILE = str(DATA_DIR / 'blacklist.json')
# DDOS_STATE_DIR moves the files a running instance writes (blacklist, traffic profile,
# logs) to another folder, so several instances can run side by side (live experiments)
_STATE_DIR = os.getenv('DDOS_STATE_DIR')
if _STATE_DIR:
    LOG_DIR = Path(_STATE_DIR) / 'logs'
    BLACKLIST_FILE = str(Path(_STATE_DIR) / 'blacklist.json')

# System Configuration
PLATFORM = platform.system().lower()  # 'linux' or 'windows'
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Network Configuration
DEFAULT_INTERFACE = 'eth0' if PLATFORM == 'linux' else 'Ethernet'
NETWORK_INTERFACE = os.getenv('INTERFACE', DEFAULT_INTERFACE)

# eBPF Configuration
EBPF_PROGRAM_PATH = str(SRC_DIR / 'ebpf' / 'xdp_filter.c')  # BCC compiles from C source
XDP_MODE = 'native'  # Options: 'native', 'generic', 'offload'

# Detection Thresholds
class DetectionThresholds:
    # Packet rate thresholds (packets per second)
    # NOTE: Lowered for testing - increase in production
    NORMAL_PPS_BASELINE = 100   # Expected normal traffic
    ALERT_PPS_THRESHOLD = 500   # Alert threshold (was 100000)
    ATTACK_PPS_THRESHOLD = 2000 # Definite attack threshold (was 500000)
    
    # Statistical thresholds
    SIGMA_MULTIPLIER = 2.0  # Standard deviation multiplier for anomaly (was 3.5)
    MIN_ENTROPY = 3.0  # Minimum source IP entropy (lower = more concentrated)
    
    # Rate of change detection
    MAX_CHANGE_RATE = 3.0  # Max allowed rate increase (3x, was 5x)

    # Protocol-mix and source-concentration checks need enough packets
    # in the interval to be meaningful
    MIN_PACKETS_FOR_SHAPE_ANALYSIS = 100

    # Mitigation: during an anomaly, block top sources that sent more than
    # this many packets in the current interval
    BLACKLIST_MIN_PACKETS = 15
    
    # Protocol distribution (approximate normal ratios)
    NORMAL_TCP_RATIO = 0.85  # 85% TCP
    NORMAL_UDP_RATIO = 0.10  # 10% UDP
    NORMAL_ICMP_RATIO = 0.05  # 5% ICMP
    PROTOCOL_DEVIATION_THRESHOLD = 0.3  # 30% deviation is suspicious

# Time Window Configuration
class TimeWindows:
    BASELINE_WINDOW = 300  # 5 minutes for baseline calculation
    DETECTION_WINDOW = 10   # 10 seconds for attack detection
    ALERT_COOLDOWN = 60     # 60 seconds between duplicate alerts
    STATISTICS_UPDATE = 1   # 1 second for statistics update

# eBPF Map Configuration
class EbpfMapConfig:
    FLOW_MAP_SIZE = 65536       # Max concurrent flows
    IP_TRACKING_SIZE = 131072   # Max tracked IPs
    BLACKLIST_SIZE = 10000      # Max blacklisted IPs
    SIGNATURE_MAP_SIZE = 1000   # Max attack signatures

# Monitoring Configuration
class MonitoringConfig:
    DASHBOARD_ENABLED = True
    DASHBOARD_PORT = 5000
    DASHBOARD_HOST = '127.0.0.1'  # pass --host 0.0.0.0 to expose on the network
    
    # Metrics export
    METRICS_ENABLED = True
    METRICS_PORT = 9090
    
    # Logging
    LOG_LEVEL = 'INFO' if not DEBUG_MODE else 'DEBUG'
    LOG_FILE = str(LOG_DIR / 'ddos_mitigation.log')
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT = 5

# Alert Configuration
class AlertConfig:
    ALERT_TO_CONSOLE = True
    ALERT_TO_FILE = True
    ALERT_FILE = str(LOG_DIR / 'alerts.log')
    
    # Future: Email, Slack, etc.
    # ALERT_EMAIL = None
    # ALERT_SLACK_WEBHOOK = None

# Traffic Profiling
class ProfilingConfig:
    ENABLE_PROFILING = True
    PROFILE_SAVE_INTERVAL = 300  # Save profile every 5 minutes
    PROFILE_FILE = str((Path(_STATE_DIR) if _STATE_DIR else DATA_DIR) / 'traffic_profile.json')
    LEARNING_PERIOD = 3600  # 1 hour initial learning period

# Performance Configuration
class PerformanceConfig:
    # eBPF processing
    EBPF_POLL_TIMEOUT = 100  # milliseconds
    
    # User-space processing
    WORKER_THREADS = 4
    BATCH_SIZE = 1000  # Process statistics in batches
    
    # Memory limits
    MAX_MEMORY_MB = 1024  # 1 GB max memory usage

# Simulation Configuration (for testing)
class SimulationConfig:
    ENABLE_SIMULATOR = False
    NORMAL_TRAFFIC_RATE = 50000  # pps
    ATTACK_TRAFFIC_RATE = 1000000  # pps
    SIMULATION_DURATION = 300  # seconds


# Phase 2: ML Configuration
class MLConfig:
    # Model paths
    MODEL_DIR = str(MODELS_DIR)
    DEFAULT_MODEL_PATH = str(MODELS_DIR / 'ddos_classifier.joblib')
    DEFAULT_REPORT_PATH = str(MODELS_DIR / 'ddos_classifier_report.json')
    
    # Dataset paths
    DATASET_DIR = str(DATA_DIR / 'cic-ddos-2019')
    
    # Training settings
    MAX_TRAINING_FILES = 5
    SAMPLES_PER_FILE = 50000
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # Model hyperparameters
    MODEL_TYPE = 'random_forest'
    N_ESTIMATORS = 100
    MAX_DEPTH = 15
    
    # Inference settings
    ML_CONFIDENCE_THRESHOLD = 70.0  # Minimum confidence for ML detection
    HYBRID_SCORE_THRESHOLD = 60.0   # Combined score threshold
    
    # Feature extraction
    FEATURE_WINDOW_SIZE = 10.0  # seconds

# Working directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
