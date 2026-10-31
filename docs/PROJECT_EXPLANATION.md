# Rapid-Corona: Complete Project Explanation

A hybrid **eBPF/XDP + Machine Learning DDoS detection and mitigation system**.
This document describes the project as it is implemented in this repository
(verified against the code and by running it), not as originally planned.

---

## 1. Purpose and objectives

A DDoS attack floods a server with more traffic than it can handle. This
project aims to:

1. **Drop attack traffic as early as possible.** An eBPF program attached with
   XDP (eXpress Data Path) runs inside the Linux network driver path and drops
   packets from blacklisted source IPs before the kernel network stack spends
   any work on them.
2. **Detect attacks automatically.** A user-space Python controller reads the
   kernel's traffic counters every second and decides whether the traffic is
   an attack, using statistical rules plus a Random Forest classifier.
3. **Mitigate automatically.** When an attack is detected, the heaviest
   sources are written into the kernel blacklist map, so the XDP program drops
   them from then on.
4. **Make it observable and demonstrable.** A web dashboard shows live status,
   and a simulation mode lets the whole pipeline run on Windows (no eBPF).

---

## 2. Repository layout

```
rapid-corona/
├── capstone/   Capstone course documents (proposal, rubrics, thesis format,
│               circular). Not part of the software; never modified.
├── project/    The runnable software (everything below)
└── other/      Non-code material: reports, diagrams, research papers, reviews
```

### 2.1 `project/` — the software

| Path | Purpose |
|---|---|
| `main.py` | Command-line entry point. Parses options, sets up logging, optionally trains a model, starts the dashboard thread and the monitoring system. |
| `requirements.txt` | Python dependencies. |
| `pytest.ini` | Test configuration (`testpaths = tests`, `pythonpath = .`). |
| `src/config.py` | **All configuration**: absolute paths (derived from the file location), detection thresholds, time windows, dashboard host/port, ML settings. |
| `src/core/system.py` | `DDoSMitigationSystem` — the orchestrator. Owns all components; `process_sample()` runs one interval through detect → baseline → mitigate → alert. Also `setup_logging()`. |
| `src/core/traffic_monitor.py` | Interface to the kernel: compiles/attaches the XDP program via BCC, reads eBPF maps, converts lifetime per-IP counters into per-interval deltas, manages the blacklist map and its JSON persistence. In simulation mode it keeps an in-memory blacklist instead. |
| `src/core/anomaly_detector.py` | `AnomalyDetector` (statistical rules + learned baseline) and `MLEnhancedAnomalyDetector` (adds the classifier and the hybrid decision logic). |
| `src/core/traffic_profiler.py` | Long-term traffic profile (average/peak pps, protocol ratios) saved to `data/traffic_profile.json`. Informational; shown on the dashboard. |
| `src/core/alert_system.py` | Alerts to console (coloured), to `logs/alerts.log` (one JSON object per line), and to an in-memory history used by the dashboard. |
| `src/core/metrics_collector.py` | CPU / memory metrics (psutil) for status logging. |
| `src/dashboard/app.py` | Flask **backend**: serves the page and the JSON API (`GET /api/status`, `POST /api/simulate_attack`), with input validation and restricted CORS. |
| `src/dashboard/static/index.html` | **Frontend**: one HTML page with CSS and vanilla JavaScript that polls `/api/status` every 5 s and renders the cards. |
| `src/ml/feature_extractor.py` | Turns per-second counters into **16 window-level features** (sliding 10 s window). Used identically in training and at runtime. |
| `src/ml/traffic_dataset.py` | Generates **labeled synthetic traffic runs** (3 benign and 8 attack scenarios) and converts them to feature windows for training/evaluation. |
| `src/ml/ml_classifier.py` | `DDoSClassifier`: Random Forest wrapper — train, predict, attack-type inference, feature importance, save/load (joblib) with version metadata. |
| `src/ml/model_trainer.py` | Training CLI: `train`, `evaluate`, `benchmark`. Grouped cross-validation, independent test set, end-to-end detector comparison, JSON reports. |
| `src/ml/data_loader.py` | Loader for the CIC-DDoS-2019 CSV dataset (per-flow features). Kept for offline experiments only — see §6.5. |
| `src/ebpf/xdp_filter.c` | **The kernel program used at runtime** (BCC dialect, compiled by BCC when `main.py` starts). |
| `src/ebpf/xdp_filter_libbpf.c`, `xdp_maps.h`, `Makefile`, `xdp_filter.o` | Standalone libbpf version of the original program (blacklist + counters only, without the P4 maps), compiled with `make` (clang). Not loaded by `main.py`. |
| `src/mitigation/` | Research extension: mitigation policies (which sources to block or throttle), the per-source attacker scorer, and the evaluation harness measuring attack traffic stopped vs legitimate traffic lost (§6.7). Evaluated offline; not yet wired into the live system. |
| `scripts/mitigation_experiment.py` | Runs the mitigation experiment (tuning on seeds 900/901, testing on 1001–1005 under three realism settings) → `data/results/mitigation_results.json`, `data/models/source_scorer.joblib`. |
| `scripts/make_figures.py` | Builds the report figures and tables in `docs/figures/` from the saved result files. |
| `scripts/run_attack_demo.py` | Demo client: posts 46 seconds of simulated traffic (normal → UDP flood → SYN flood → multi-vector) to a running system. |
| `scripts/attack_simulator.py` | Sends **real packets** (UDP/TCP/ICMP/HTTP, optional spoofed sources via Scapy). For Linux lab testing with a veth pair. |
| `scripts/benchmark.py` | Times one monitoring interval (`process_sample`) in user space for the statistical and hybrid detectors and both policies. |
| `scripts/safety_experiment.py` | Mitigation-safety scenarios: flash crowds, legitimate bulk transfers, floods with a concurrent flash crowd. |
| `scripts/verify_results.py` | Recomputes every number quoted in the paper, report and slides from the result files. |
| `scripts/reproduce_all.py` | Runs every experiment, figure, verification and the tests; writes a manifest with output hashes. |
| `scripts/setup_ebpf.sh` / `setup_ebpf.ps1` | Environment setup for Linux / Windows. |
| `tests/` | pytest suite (86 test cases, see §9). |
| `data/blacklist.json` | Persisted blacklist (live XDP mode); restored into the kernel map on start. |
| `data/traffic_profile.json` | Persisted traffic profile. |
| `data/models/ddos_classifier.joblib` | The trained model. |
| `data/models/ddos_classifier_report.json` | Training report: CV results, chosen hyperparameters, held-out test metrics, per-scenario results. |
| `data/models/ddos_classifier_eval.json` | End-to-end evaluation report (statistical vs hybrid detector). |
| `docs/` | `RUN.md` (quick start), `USAGE_GUIDE.md` (options/reference), this file. |
| `logs/` | Runtime logs (`ddos_mitigation.log`, `alerts.log`, benchmark output). Not committed. |

### 2.2 `other/` — documents

| Folder | Contents |
|---|---|
| `reports/` (35 files) | Project reports and review packages (`FINAL_REPORT_*`, `FINAL_REVIEW_*`, viva Q&A, `projectexplained.*`, `DIAGRAMS.md`, `DOCUMENTATION_INDEX.md`, …) in .md/.pdf/.html/.docx. Links inside the .md files point to `../diagrams/` and `../../project/`. |
| `diagrams/` (9 files) | Architecture/flow diagrams (.png) and `IMAGE_PROMPT.md` (the prompt used to generate the architecture image). |
| `research-papers/` | Paper drafts (`IEEE_DDoS_Paper.tex`, review documents, spreadsheets) and `references/` (9 published papers used as references). |
| `review/` | Review submissions (documents, presentations) and `supporting-material/` (reference papers, figures, paper drafts used for the review). |

These documents were written at different stages of the project; where they
disagree with the code (e.g. older ML descriptions mentioning 64 CIC
features), **the code and this file are authoritative**.

---

## 3. Technologies

| Layer | Technology | Used for |
|---|---|---|
| Kernel | **eBPF / XDP**, C | Per-packet parsing, counting, dropping |
| Kernel ↔ Python | **BCC** (`python3-bpfcc`) | Compiling the C program at runtime, attaching to the NIC, reading/writing maps |
| Build (optional) | clang/LLVM, libbpf, make | Standalone compile of `xdp_filter_libbpf.c` |
| Backend | **Python 3** (tested on 3.10.6), threading | Controller, detection, API |
| Web | **Flask 3**, flask-cors | Dashboard backend / JSON API |
| Frontend | HTML, CSS, vanilla JavaScript (fetch API) | Dashboard page |
| ML | **scikit-learn** (RandomForestClassifier, GroupKFold), NumPy, pandas, joblib | Training, inference, model persistence |
| Ops | psutil, coloredlogs | Metrics, console logging |
| Testing | pytest | Unit, integration and API tests |
| Demo / lab | requests, Scapy (optional) | Demo client, spoofed packet generation |

**There is no database.** State lives in: eBPF maps (in kernel memory), JSON
files in `data/`, the joblib model file, and log files in `logs/`.

---

## 4. Architecture and how the components interact

```
                       ┌────────────────────────── Linux kernel ─────────────────────────┐
  incoming packets ──► │ XDP program (src/ebpf/xdp_filter.c), runs for every packet:     │
                       │   parse Ethernet/IPv4/TCP/UDP                                   │
                       │   update ip_tracking_map, flow_map (per source / per flow)      │
                       │   if src in blacklist_map  → XDP_DROP                           │
                       │   if SYN and src syn_count > 1000 → XDP_DROP                    │
                       │   else XDP_PASS ; update stats_map (per-CPU totals)             │
                       └────────────▲───────────────────────────────┬────────────────────┘
                    write blacklist │                               │ read maps every 1 s
                                    │                               ▼
┌──────────────────────────── Python process (main.py) ──────────────────────────────────────┐
│  TrafficMonitor ── stats + per-IP deltas ──► DDoSMitigationSystem.process_sample()          │
│                                                ├─ AnomalyDetector (statistical score)        │
│                                                ├─ FeatureExtractor ─► DDoSClassifier (ML)    │
│                                                ├─ hybrid decision                            │
│                                                ├─ learn baseline (normal intervals only)     │
│                                                ├─ TrafficProfiler                            │
│                                                ├─ attack? blacklist top sources ─────────────┼─► kernel map + data/blacklist.json
│                                                └─ attack? AlertSystem ───────────────────────┼─► console, logs/alerts.log
│                                                                                              │
│  Flask thread: GET /api/status ◄── get_status()          POST /api/simulate_attack ──► process_sample()
└───────────▲──────────────────────────────────────────────────────▲───────────────────────────┘
            │ every 5 s (fetch)                                    │ 1 sample per simulated second
   Browser: static/index.html                            scripts/run_attack_demo.py
```

Key point: **live kernel samples and simulated demo samples go through exactly
the same `process_sample()` code path.** The only difference is where the
numbers come from.

The monitoring loop thread and the Flask thread both call into the system, so
`process_sample()` and `get_status()` are serialized with a re-entrant lock.

---

## 5. Execution flow, step by step

### 5.1 Startup (`main.py`)
1. Parse arguments (`--interface`, `--mode`, `--dashboard`, `--host`, `--port`,
   `--ml-model`, `--train-model`, `--enable-sim-api`, `--debug`).
2. `setup_logging()`: UTF-8 file handler → `logs/ddos_mitigation.log`, coloured console.
3. On Linux, exit unless running as root (needed to load XDP programs).
4. If `--train-model`: train a new model with the default training settings.
5. Create `DDoSMitigationSystem`:
   - If no `--ml-model` is given and `data/models/ddos_classifier.joblib`
     exists, it is auto-detected → `MLEnhancedAnomalyDetector`.
   - The model is loaded and checked: every feature it expects must be
     producible by the runtime `FeatureExtractor`, otherwise ML is disabled
     with an error explaining how to retrain.
6. If `--dashboard`: start Flask in a daemon thread (default `127.0.0.1:5000`).
7. `system.start()`:
   - Register SIGINT/SIGTERM handlers (clean unload of the XDP program).
   - If BCC is unavailable: on Windows → **simulation mode**; on Linux → error.
   - Live mode: `TrafficMonitor.load_xdp_program()` reads `xdp_filter.c`, BCC
     compiles it, attaches it in native/generic/offload mode, and restores the
     saved blacklist into `blacklist_map`.
   - Enter the monitoring loop.

### 5.2 Monitoring loop (every 1 second)
- **Live mode**: read `stats_map` (summed over CPUs) and `ip_tracking_map`.
  Per-IP counters in the kernel are lifetime totals, so the monitor subtracts
  the previous reading to get **what each source sent in this interval**
  (top 1000 sources). Then `process_sample(stats, ip_deltas)`.
- **Simulation mode**: no kernel data; the loop only collects metrics. Samples
  arrive through `POST /api/simulate_attack`.
- Every 10 iterations a status line is logged.

### 5.3 `process_sample()` — one interval
1. Compute current pps/bps from the cumulative counters and elapsed time.
2. **Detect** (before updating the baseline, so the new interval is compared
   with the past): statistical score + ML prediction → combined verdict (§6).
3. **Learn**: add the interval to the baseline history *only if it was not an
   attack*, so floods do not inflate the "normal" baseline.
4. Update the traffic profile; remember the top 10 sources for the dashboard.
5. **Mitigate** (policy chosen with `--policy`):
   - `p4` (default): every interval goes to `src/mitigation/live.py`, which runs
     the evaluated source-scoring policy (§6.7). Normal intervals build each
     source's history; in attack intervals confident attackers are blocked for
     30 s, uncertain ones are rate-limited for 30 s, and during spoofed floods
     unknown sources share a small budget. The decisions are written to kernel
     maps: `blacklist_map` (value = expiry time, 0 = permanent), `ratelimit_map`
     (token bucket), `known_map` and `guard_map` (spoofing fallback). In
     simulation mode `TrafficMonitor.simulate_drops()` applies the same rules.
   - `top5` (original rule): for the 5 heaviest sources of this interval with
     more than 15 packets (`BLACKLIST_MIN_PACKETS`), skip already-blocked IPs
     and block the rest permanently (also saved to `data/blacklist.json`).
   Invalid, loopback, multicast and broadcast addresses are always refused.
6. If attack:
   - **Alert** (at most one per 60 s, `ALERT_COOLDOWN`): console + `logs/alerts.log`
     + in-memory history, including the reasons and metrics.

### 5.4 Dashboard
`static/index.html` fetches `/api/status` every 5 s. `get_status()` returns:
interface and mode, counters, current/baseline pps, learned baseline,
profile, blacklist, latest top sources, last detection (score, source, type,
reasons), recent alerts, ML statistics (predictions, attacks, average
inference time, model accuracy from the held-out test) and feature
importances. All untrusted strings are HTML-escaped before rendering.

### 5.5 Demo (`scripts/run_attack_demo.py`)
Builds cumulative counters and per-IP lists and posts one sample per simulated
second (`interval_seconds: 1`, so HTTP latency does not distort rates):
1. 15 s normal traffic (~50 pps, 5–20 internal IPs) — builds the baseline.
2. UDP flood ramping 1,000 → 5,000 → 10,000 pps from 20–100 bots.
3. SYN flood ~15,000 pps from 1,000 bots.
4. Multi-vector flood ~20,000 pps from 3,000 bots.

It prints the score, the detector that fired and the blacklist size per step,
and a final summary from `/api/status`.

---

## 6. Detection logic and the ML model

### 6.1 Statistical detector (`AnomalyDetector`)
All checks use the **latest interval**. Points are summed; ≥ 50 = anomaly.

| Check | Condition (defaults in `config.py`) | Points |
|---|---|---|
| Absolute rate | > 2000 pps / > 500 pps | +50 / +25 |
| Deviation from baseline | > 2 σ above learned mean (σ floored at max(σ, 10 % of mean, 1)) | +30 |
| Rate of change | > 3× the previous normal interval | +20 |
| TCP share | differs from 0.85 by > 0.3 | +15 |
| UDP share | > 0.40 | +15 |
| ICMP share | > 0.35 | +15 |
| Source entropy | Shannon entropy of sources < 3.0 bits | +20 |
| Concentration | top-10 sources > 80 % of packets | +10 |
| SYN flood | any source with > 500 SYNs in the interval | +25 |

Protocol and source-distribution checks only run when the interval has at
least 100 packets. The baseline needs 10 normal samples (history of 300).

### 6.2 Features (`FeatureExtractor`)
The XDP program provides totals, protocol counts and per-source packets/SYNs —
not per-flow timing. So the model uses **16 features over a sliding 10 s
window** (only windows with ≥ 3 samples are classified):

`pps_mean`, `pps_max`, `pps_cv` (burstiness), `bps_mean`, `avg_packet_size`,
`tcp_ratio`, `udp_ratio`, `icmp_ratio`, `syn_share` (SYNs / packets of listed
sources), `unique_src_ips`, `src_ip_entropy_norm` (0–1), `top1_share`,
`top10_share`, `packets_per_ip`, `pps_ratio_long` (window pps / 300 s median),
`pps_growth` (latest second / window mean).

### 6.3 Training (`python -m src.ml.model_trainer train`)
1. **Data**: `traffic_dataset.py` simulates traffic second by second, as the
   kernel would report it. Each run lasts 70–120 s with a normal baseline of
   5–5,000 pps (log-uniform), 5–3,000 normal clients, and an event starting at
   30–55 s (30 % of events end early so recovery is represented).
   - Benign: `normal`, `normal_bursty` (short 1.5–3.5× bursts), `flash_crowd`
     (3–15× growth from many new legitimate clients).
   - Attack: `udp_flood`, `syn_flood`, `icmp_flood`, `mixed_flood`,
     `udp_flood_spoofed`, `syn_flood_spoofed`, `low_rate_udp`,
     `low_rate_syn` (only 1.3–2.5× normal volume).
   - Each run is pushed through the same `FeatureExtractor`; a window is
     labeled attack when most of its seconds contain attack traffic.
2. **Split**: 600 training runs (seed 7) → 55,256 windows; 300 **independent**
   test runs (seed 1007) → 27,262 windows. Windows from one run are correlated,
   so cross-validation uses `GroupKFold` by run (no leakage).
3. **Model selection**: 12 Random Forest configurations (trees 100/200, depth
   8/12/unlimited, min leaf 2/5), 5-fold grouped CV on F1; the simplest model
   within one standard deviation of the best is chosen → 100 trees, unlimited
   depth, min leaf 2, balanced class weights.
4. **Evaluation**: held-out test metrics and per-scenario results are stored
   inside the model and in `ddos_classifier_report.json`.

### 6.4 Prediction at runtime
Each interval: update the extractor → if the window has ≥ 3 samples, build the
feature row **in the model's stored feature order** → `predict_proba` →
attack if P(attack) ≥ 0.5, confidence = max probability. The attack type is
inferred from the features (`syn_share > 0.5` → SYN_Flood, `udp_ratio > 0.6`
→ UDP_Flood, `icmp_ratio > 0.5` → ICMP_Flood, else DDoS_Generic).
Single-sample inference takes ~2 ms.

### 6.5 Hybrid decision (`MLEnhancedAnomalyDetector`)
In order:
1. ML attack with confidence ≥ 85 % → attack (`ml`)
2. Statistical anomaly and ML attack → attack (`hybrid`)
3. Statistical score ≥ 90 → attack even if ML disagrees (`statistical`)
4. ML attack with confidence ≥ 70 % → attack (`ml`)
5. ML attack and combined score ≥ 60 → attack (`hybrid`)
6. Otherwise normal — a moderate statistical score the classifier does not
   confirm (e.g. a flash crowd) is not acted on.

**Why not the CIC-DDoS-2019 dataset?** Its features are per-flow
(inter-arrival times, active/idle times, header lengths, …) and cannot be
computed from the counters the XDP program exports. A model trained on them
cannot be fed honestly at runtime; the original model failed on every
prediction for exactly this reason. The CIC loader remains for offline
experiments (`--source cic`), and `main.py` refuses to load such models.

### 6.6 Measured results
Held-out test windows (300 independent runs): accuracy 0.995, precision 0.990,
recall 0.990, F1 0.990, ROC-AUC 0.9999; flash-crowd and normal windows: 0 %
false positives.

End-to-end replay of 200 fresh runs (seed 3030), second by second:

| | Statistical only | Hybrid |
|---|---|---|
| Attack seconds detected | 77.2 % | 92.7 % |
| False alarms on normal seconds | 25.8 % | 3.0 % |
| Attack runs detected | 89/110 | 110/110 |
| Normal runs with ≥ 1 false alarm | 67/90 | 26/90 |
| Median detection delay | 0 s | 4 s |

These numbers are on **synthetic** traffic; they show the pipeline is
consistent and that ML adds value over the rules, not real-world accuracy.

### 6.7 Mitigation policies (research extension, `src/mitigation/`)
Detection answers "is there an attack?"; mitigation decides **which traffic to
drop**. The original rule (block the 5 heaviest sources of each anomalous
second, permanently) can block heavy legitimate users and does little against
large or spoofed botnets. The extension compares, on the same cached detector
verdicts:

| Policy | Idea |
|---|---|
| P0 Top-5 blacklist, permanent | current system rule |
| P1 Top-5 blacklist, 30 s TTL | blocks expire |
| P2 History-based filter (Peng et al., ICC 2003) | during an attack, admit only sources seen in normal times |
| P3a Aggregate rate limit | cap total admitted traffic at 1.5× the learned baseline |
| P3b Per-source rate limit | cap every source at the 99th percentile of normal per-source rates |
| AHIF (Goldstein et al., ICN 2008) | admit sources in order of their share of normal traffic until 1.5× the baseline; drop the rest |
| SIG (Dimolianis et al., 2021) | drop every packet whose header signature (TTL + TCP window, or size) a Random Forest labels malicious; SIG* = perfect labels (upper bound) |
| **P4 Source scoring (proposed)** | logistic-regression P(attacker) per source from five counter-only features (known, rate jump vs its own history, rate vs a typical source, SYN share, age); score ≥ 0.8: block 30 s; 0.7–0.8: rate-limit 30 s; otherwise untouched; when ≥ 100 brand-new sources carry ≥ 50% of the traffic (spoofing), unknown sources share what remains of a 1.5× baseline budget after known sources. **Runs in the live system** (default policy). |

Ground truth comes from the generator, which now records a stable identity
and attacker/legitimate label for every source. Two opt-in realism options
make history-based decisions non-trivial: `client_churn` (new legitimate
users keep arriving) and `bot_pre_activity` (bots send normal-looking
traffic before the attack). Defaults are off, so the detection training
data and model evaluation are unchanged (verified by fingerprint).

Protocol: scorer and SIG classifier trained on seed 900, P4 thresholds chosen
on seed 901 (maximize attack-dropped % − 2 × legitimate-dropped %), results on
unseen seeds 2001–2005 (100 runs each) for three realism settings
(`scripts/mitigation_experiment.py`). Seeds 1001–1005 were retired after the
first P4 version was fixed.

Real legitimate traffic: `scripts/mawi_extract.py` streams the first 180 s of
a public MAWI backbone trace; `scripts/trace_experiment.py` lets the 20 busiest
destinations play the server and injects attacks on top, with nothing tuned on
the trace. Results: `docs/figures/RESULTS_TABLES.md`, `fig06`–`fig11`.

| Realistic setting / real trace | Attack dropped | Legit dropped |
|---|---|---|
| P0 top-5 | 34.5% / 32.3% | 27.8% / 7.0% |
| AHIF | 92.5% / 71.9% | 9.9% / 23.4% |
| SIG (trained) | 71.1% / 69.6% | 3.2% / 24.2% |
| SIG* (perfect labels) | 98.2% / 89.1% | 1.8% / 0.3% |
| **P4** | **89.6% / 52.9%** | **2.7% / 1.8%** |

---

## 7. Installation and running

All commands from the `project/` folder.

```bash
pip install -r requirements.txt
```

**Windows (simulation mode)**
```bash
python main.py --interface Ethernet --dashboard
python scripts/run_attack_demo.py        # second terminal
```
Open http://127.0.0.1:5000.

**Linux (real XDP)** — needs root, kernel ≥ 4.18 and BCC:
```bash
sudo ./scripts/setup_ebpf.sh             # or: sudo apt-get install python3-bpfcc
sudo python3 main.py --interface eth0 --mode generic --dashboard
```
Single-machine lab with a veth pair (victim `veth1`, attacker `veth0`):
```bash
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 up && sudo ip link set veth1 up
sudo ip addr add 192.168.99.1/24 dev veth0 && sudo ip addr add 192.168.99.2/24 dev veth1
sudo python3 main.py --interface veth1 --mode generic --dashboard
sudo python3 scripts/attack_simulator.py --type mixed --distributed --target 192.168.99.2 --duration 30 --interface veth0
```

**Configuration** lives in `src/config.py`: thresholds (`DetectionThresholds`),
intervals and cooldowns (`TimeWindows`), dashboard (`MonitoringConfig`),
paths and ML settings (`MLConfig`). Environment variables: `INTERFACE`,
`DEBUG=true`.

**Model commands**
```bash
python -m src.ml.model_trainer train      # ~2-3 minutes
python -m src.ml.model_trainer evaluate --seed 3030 --report data/models/ddos_classifier_eval.json
python -m src.ml.model_trainer benchmark
```

---

## 8. API reference

| Method / path | Description |
|---|---|
| `GET /` | Dashboard page (`static/index.html`) |
| `GET /api/status` | JSON status (CORS enabled, read-only) |
| `POST /api/simulate_attack` | Body: `{"stats": {cumulative counters}, "ip_stats": [{"ip", "packets", "syn_count", ...}], "label", "interval_seconds"}`. Requires `Content-Type: application/json`. Enabled in simulation mode, or in live mode only with `--enable-sim-api`. Returns the verdict, score, detector, attack type, whether an alert fired, baseline and blacklist size. Returns 400 for invalid input, 403 when disabled. |

---

## 9. Testing

```bash
python -m pytest             # 86 test cases, ~15 s
```

| File | What it covers |
|---|---|
| `test_detector.py` | Baseline learning, rate/protocol/entropy rules, cooldown, regressions (detect-before-learn, interval ratios, low-volume gating, traffic drops) and the hybrid decision rules |
| `test_ml_classifier.py` | Data loader, feature extractor (window maths, expiry, counter resets, column selection), classifier train/predict/save/load, latency |
| `test_trained_model.py` | The shipped model: compatibility with the extractor, stored evaluation, accuracy on unseen runs, Python types, latency; rejection of per-flow models; dataset determinism and labels |
| `test_system.py` | End-to-end `process_sample` (detection, blacklisting, alerts, baseline not poisoned, idle host never flagged), blacklist validation, per-IP delta computation, dashboard API (validation, 403/400 paths, CORS, escaping) |
| `test_layout.py` | All configured paths are absolute and inside `project/`; required assets exist; components work from another working directory |

Manual end-to-end check: start `main.py --dashboard`, run
`scripts/run_attack_demo.py`, confirm the flood is detected, IPs are
blacklisted, an alert appears and the dashboard updates.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run `-m` commands and pytest from `project/` (or use `python project/main.py`). |
| `BCC/eBPF not available!` (Linux) | Install `python3-bpfcc` and kernel headers; run with `sudo`. |
| `This program must be run with sudo/root privileges` | Loading XDP needs root on Linux. |
| `Failed to load XDP program` | Driver lacks native XDP → use `--mode generic`. |
| Dashboard/demo very slow on Windows | `localhost` tries IPv6 first (~2 s per request); use `127.0.0.1` (the default now). |
| `Address already in use` | Another instance is running; stop it or use `--port`. |
| `/api/simulate_attack` returns 403 | Live XDP mode; start with `--enable-sim-api` to allow injection. |
| "ML model is incompatible with the runtime feature extractor" | Model trained on per-flow (CIC) features; run `python -m src.ml.model_trainer train`. |
| scikit-learn version warning on model load | Model saved with another version; retrain. |
| `make` fails with `asm/types.h` not found | Install `libbpf-dev`, `clang`; the Makefile adds the multiarch include path. |
| No dashboard data in simulation mode | Expected until `run_attack_demo.py` (or another client) posts samples. |

---

## 11. Issues found and fixed during the review

1. **Windows demo detected nothing** — the loop reset detector state every
   second; now one shared `process_sample()` path.
2. **Live detection always saw 0 pps** — baseline was updated before detection;
   order fixed, attacks no longer learned into the baseline.
3. **Lifetime per-IP counters** kept idle hosts "under attack" and re-blacklisted
   them every second; now per-interval deltas.
4. **ML never ran** — 12-feature model vs 64-feature extractor, error hidden at
   debug level; extractor rewritten, model retrained, compatibility enforced.
5. **Hybrid logic could never veto** a statistical detection; fixed to match its
   documented rules.
6. **Security**: unauthenticated cross-origin injection API that could blacklist
   any IP, dashboard bound to all interfaces, unescaped HTML; all fixed.
7. Lost file logging, Unicode crashes, wrong demo phase output, hard-coded
   "95.3 %" accuracy, alphabetical feature-importance display, ~50 ms ML
   inference, slow simulators, import-time crash for non-root Linux users,
   Debian build failure of the eBPF Makefile.
8. **Reorganization**: code moved into `project/` (packages `core`, `dashboard`,
   `ml`, `ebpf`, `simulation`; scripts and docs separated; frontend extracted
   from Python into `static/index.html`), documents moved into `other/`, all
   paths centralized in `config.py` as absolute paths.

---

## 12. Known limitations and suggested improvements

- **Live XDP path not verified in this environment** (no root available). The
  libbpf variant compiles cleanly; the BCC variant used by `main.py` must be
  tested on a Linux machine with root.
- **Kernel program**:
  - The SYN rule (`syn_count > 1000`) uses a lifetime counter, so a busy
    legitimate client can be SYN-blocked permanently. Suggest a per-second
    window in the map entry.
  - `flow_count` is never incremented. (The unused `config_map` and `update_config()` were removed.)
  - `xdp_filter.c` uses plain hash maps (the libbpf version uses LRU): once
    131,072 sources are tracked, new sources are no longer counted (spoofed
    floods can fill it). Suggest `BPF_TABLE("lru_hash", ...)`.
  - IPv6 traffic is passed without inspection.
- **ML is trained on synthetic data only.** Validate on captured traffic, e.g. by
  replaying pcaps through the XDP program and recording the per-second counters.
- **Thresholds are demo values** (500/2000 pps). Tune per network; with them,
  some normal high-volume runs still raise a false alarm, which blacklists
  legitimate sources.
- **Blacklist never expires**; there is no dashboard control to unblock an IP.
- The dashboard runs on Flask's development server with no authentication
  (bound to localhost by default).
- Attack-type labels lag behind attack changes by up to the 10 s window.
- The simulation-mode blacklist is in memory only (not persisted).
