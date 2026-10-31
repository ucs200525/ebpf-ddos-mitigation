# DDoS Mitigation System - Usage Guide

## Overview

A machine learning-based DDoS mitigation system with real-time traffic analysis and automated defense.

- **Phase 1**: Statistical anomaly detection with eBPF/XDP
- **Phase 2**: ML-based classification using CIC-DDoS-2019 trained models

---

## Prerequisites

### Linux
- Linux kernel 4.18+ with XDP support
- Ubuntu 20.04+, RHEL 8+, or compatible
- Root/sudo access
- Python 3.8+

### Dependencies
```bash
pip3 install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `scipy`, `flask`, `scikit-learn`, `joblib`, `psutil`, `coloredlogs`

---

## Quick Start


### 1. Install & Setup

```bash
# Navigate to the project folder (all commands below run from here)
cd rapid-corona/project

# Run setup (Linux)
sudo ./scripts/setup_ebpf.sh

# Compile eBPF programs
cd src/ebpf && make && cd ../..

# Install Python dependencies
pip3 install -r requirements.txt
```

### 2. Run the System

```bash
# Basic - Statistical detection only (Phase 1)
sudo python3 main.py --interface enp0s3 --dashboard

# With ML classification (Phase 2)
sudo python3 main.py --interface enp0s3 --train-model --dashboard

# With pre-trained ML model
sudo python3 main.py --interface enp0s3 --ml-model data/models/ddos_classifier.joblib --dashboard
```

### 3. Access Dashboard
Open **http://localhost:5000** in your browser.

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--interface, -i` | Network interface to monitor | eth0 |
| `--mode, -m` | XDP mode: native, generic, offload | native |
| `--dashboard` | Enable web dashboard | False |
| `--port, -p` | Dashboard port | 5000 |
| `--host` | Dashboard bind address (`0.0.0.0` exposes it on the network) | 127.0.0.1 |
| `--enable-sim-api` | Allow `/api/simulate_attack` on a live XDP interface (always on in simulation mode) | False |
| `--debug, -d` | Enable debug logging | False |
| `--ml-model` | Path to trained ML model | None |
| `--train-model` | Train ML model before starting | False |
| `--data-path` | Path to CIC-DDoS-2019 dataset (only for `model_trainer --source cic`) | data/cic-ddos-2019 |

### Examples

```bash
# Generic XDP mode (works on all interfaces)
sudo python3 main.py --interface enp0s3 --mode generic --dashboard

# With specific dashboard port
sudo python3 main.py --interface eth0 --dashboard --port 8080

# Debug mode with ML
sudo python3 main.py --interface eth0 --ml-model data/models/ddos_classifier.joblib --debug
```

---

## Phase 2: ML Classification

### How the model sees traffic

The XDP program only exports aggregate counters and per-source-IP counters
(packets, bytes, SYN count). It cannot provide CIC-DDoS-2019 per-flow features
(inter-arrival times, active/idle times, ...), so the runtime model works on
**16 window-level features** computed over the last 10 seconds by
`src/ml/feature_extractor.py`: packet/byte rates, burstiness, protocol mix,
SYN share, source-IP count/entropy/concentration, and rate relative to the
long-term median.

The shipped model (`data/models/ddos_classifier.joblib`) is a Random Forest
trained on **synthetic** labeled traffic runs (`src/ml/traffic_dataset.py`)
pushed through that same extractor, so training and inference features are
identical. Scenarios include hard negatives (bursty traffic, flash crowds) and
hard positives (low-rate floods, spoofed-source floods).

### Training and evaluating

```bash
# Train (CV-grouped hyperparameter search + independent held-out test runs)
python3 -m src.ml.model_trainer train
#   -> data/models/ddos_classifier.joblib
#   -> data/models/ddos_classifier_report.json (test metrics, per-scenario results)

# Evaluate on fresh runs, and compare statistical-only vs hybrid detection
python3 -m src.ml.model_trainer evaluate --seed 3030 --report data/models/ddos_classifier_eval.json

# Benchmark inference speed
python3 -m src.ml.model_trainer benchmark
```

### Training on real attacks (CIC-DDoS2019)

`run.bat cic <folder>` (or `bash run.sh cic <folder>`) trains and tests all three levels
on the dataset's raw packet captures, using the same feature code as the running system.
`main.py --models cic` then uses those models. See `docs/REAL_DATA.md` for the steps,
the labels and what this dataset can and cannot show.

`python -m src.ml.model_trainer train --source cic --data-path data/cic-ddos-2019/` is the
older path: it trains on the CICFlowMeter CSV columns with a random split. Those columns
(bidirectional, CICFlowMeter-specific) cannot be computed by XDP, so `main.py` refuses that
model. The level-3 flow model above uses only inbound counters that the XDP flow map keeps.

### Attack types reported

Binary attack/benign classification; for attacks a type is inferred from the
features: `SYN_Flood`, `UDP_Flood`, `ICMP_Flood`, or `DDoS_Generic`.

---

## Detection Methods

### Phase 1: Statistical Detection

All checks run on the **latest interval** (deltas of the kernel counters), not
lifetime totals.

1. **Absolute Thresholds** - PPS above `ALERT_PPS_THRESHOLD` / `ATTACK_PPS_THRESHOLD`
   (500 / 2000 in `config.py`, set low for demos; **tune these for your network**)
2. **Statistical Deviation** - PPS more than 2 sigma above the learned baseline
3. **Rate of Change** - more than 3x the previous interval
4. **Protocol Distribution** - TCP/UDP/ICMP mix of the interval
5. **IP Entropy / Concentration** - few sources carrying most packets
6. **SYN Flood Detection** - sources with > 500 SYNs in the interval

Checks 4-5 only run when the interval has at least
`MIN_PACKETS_FOR_SHAPE_ANALYSIS` (100) packets. Intervals flagged as attacks are
not learned into the baseline.

### Phase 2: ML Classification

- **Random Forest** on 16 window-level features (see above)
- **Hybrid detection** combining statistical + ML verdicts

### Hybrid Scoring

- ML attack with confidence >= 85% → attack (`ml`)
- Statistical anomaly and ML attack → attack (`hybrid`)
- Statistical score >= 90 → attack even if ML disagrees (`statistical`)
- ML attack with confidence >= 70% → attack (`ml`)
- Otherwise → normal (a moderate statistical score the classifier does not
  confirm, e.g. a flash crowd, is not acted on)

### Mitigation

During an attack interval the top 5 sources sending more than
`BLACKLIST_MIN_PACKETS` (15) packets in that interval are added to the XDP
blacklist map. Loopback, multicast, broadcast and malformed addresses are never
blacklisted.

---

## Web Dashboard

Access at **http://localhost:5000**

### Dashboard Cards

| Card | Information |
|------|-------------|
| Real-time Status | Interface, current PPS, baseline PPS, drop rate |
| Traffic Statistics | Total packets/bytes, dropped/passed counts |
| Protocol Distribution | TCP, UDP, ICMP, Other packet counts |
| Baseline Profile | Mean PPS, std dev, samples, learning status |
| Top IPs | Highest traffic source IPs |
| Blacklist | Blocked IP addresses |
| **ML Classification** | Model accuracy, predictions, attacks detected, inference time |
| **Feature Importance** | Top features contributing to ML predictions |
| Recent Alerts | Alert history with severity |

---

## Configuration

Edit `config.py` to customize:

### Detection Thresholds
```python
class DetectionThresholds:
    ALERT_PPS_THRESHOLD = 500      # Alert threshold
    ATTACK_PPS_THRESHOLD = 2000    # Attack threshold
    SIGMA_MULTIPLIER = 2.0         # Statistical deviation
    MIN_ENTROPY = 3.0              # IP diversity threshold
```

### ML Configuration
```python
class MLConfig:
    DEFAULT_MODEL_PATH = 'data/models/ddos_classifier.joblib'
    ML_CONFIDENCE_THRESHOLD = 70.0   # Min confidence for ML detection
    HYBRID_SCORE_THRESHOLD = 60.0    # Combined score threshold
    N_ESTIMATORS = 100               # Random Forest trees
    MAX_DEPTH = 15                   # Tree depth
```

### Time Windows
```python
class TimeWindows:
    BASELINE_WINDOW = 300   # 5 min baseline learning
    DETECTION_WINDOW = 10   # 10 sec detection window
    ALERT_COOLDOWN = 60     # 60 sec between alerts
```

---

## Testing

### Run Unit Tests
```bash
# All tests
python3 -m pytest tests/ -v

# ML classifier tests
python3 -m pytest tests/test_ml_classifier.py -v

# Anomaly detector tests
python3 -m pytest tests/test_detector.py -v
```

### Benchmark Performance
```bash
# User-space time of one monitoring interval (detection + mitigation), simulation mode
python3 scripts/benchmark.py

# In-kernel XDP cost per packet, control-plane CPU/RSS, detection and mitigation delay
# (Linux, root): written to logs/live_demo_<time>/live_metrics.json
sudo bash scripts/live_xdp_demo.sh
```

### Reproduce every result
```bash
python3 scripts/reproduce_all.py        # experiments, figures, verification, tests
```

---

## Traffic Simulation

### Generate Attack Traffic
```bash
# Sends real packets; use only against machines/networks you own
python3 scripts/attack_simulator.py --type udp --target 192.168.99.2 --pps 10000 --duration 60
python3 scripts/attack_simulator.py --type tcp --target 192.168.99.2 --port 80 --pps 5000 --duration 30
```

To exercise detection without sending packets (works on Windows), run
`python scripts/run_attack_demo.py` against a running `main.py --dashboard`.

### Attack Types (`--type`)
- `udp` - High-volume UDP packets
- `tcp` - TCP connection (SYN) flood
- `icmp` - ICMP ping flood (root)
- `http` - HTTP GET flood
- `mixed` - Multi-vector attack
- `spike` - Volumetric ramp up / sustain / ramp down

Add `--distributed` (Linux, root, scapy) to spoof random botnet source IPs.

---

## Files & Directories

See the structure in [../README.md](../README.md#-project-structure) and the
full walkthrough in [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).

---

## Troubleshooting

### eBPF Won't Load
```bash
# Check kernel version (need 4.18+)
uname -r

# Try generic mode
sudo python3 main.py --interface eth0 --mode generic
```

### Permission Denied
```bash
# Require root for XDP
sudo python3 main.py --interface eth0
```

### ML Model Not Loading
```bash
# Ensure model exists
ls -la data/models/

# Train a new model
python3 -m src.ml.model_trainer train
```

If the log says the model is "incompatible with the runtime feature extractor",
it was trained on per-flow (CIC) features; retrain with the command above.

### Missing Dependencies
```bash
pip3 install scikit-learn joblib
```

---

## Performance Expectations

| Metric | Target |
|--------|--------|
| Packet handling | 5M+ pps |
| Detection latency | <1 second |
| ML inference | <10 ms per prediction |
| CPU overhead | <20% at 5M pps |
| Drop rate | >10M pps (blacklisted) |

---

## Log Files

| File | Content |
|------|---------|
| `logs/ddos_mitigation.log` | Main application log |
| `logs/alerts.log` | Alert history |
| `data/traffic_profile.json` | Learned traffic baseline |
| `data/models/*.joblib` | Trained ML models |

---

## License

MIT License
