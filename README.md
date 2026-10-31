# 🛡️ Hybrid XDP & ML-Based DDoS Mitigation System

This project is a high-performance **DDoS detection and mitigation system** that combines the speed of **eBPF/XDP** (Express Data Path) for kernel-level packet filtering with the intelligence of **Machine Learning (Random Forest)** for traffic classification.

By leveraging XDP, the system can block malicious traffic **before it even reaches the operating system's network stack**, achieving extremely high throughput and low latency.

---

## 🚀 Key Features

### 1. **Kernel-Level Blocking (XDP)**
   - Uses **eBPF (Extended Berkeley Packet Filter)** to hook into the network driver.
   - Drops malicious packets directly at the NIC level.
   - Extremely low CPU overhead compared to userspace firewalls (iptables/nftables).
   - Blocks based on a dynamic **Blacklist Map** shared between kernel and userspace.

### 2. **Intelligent Anomaly Detection**
   - **Baseline Profiling**: Automatically learns normal traffic patterns (PPS, Protocol distribution) during a learning phase.
   - **ML Classification**: Uses a trained **Random Forest Classifier** on 16 window-level traffic features (rates, protocol mix, SYN share, source-IP entropy/concentration) computed from the eBPF counters. The shipped model is trained **only on synthetic labelled traffic**; its accuracy (0.995 on held-out synthetic windows) is not a measurement on real attacks. See `docs/REPRODUCIBILITY.md` for what each result does and does not show.
   - **Hybrid Approach**: Combines statistical thresholds (for volumetric floods) with ML (for subtle L7/protocol attacks).
   - **Three levels, also trained on real attacks**: level 1 (is there an attack: per-second
     windows), level 2 (which sources: per-source scorer, P4) and level 3 (does this source's
     traffic look like an attack: per-flow classifier on XDP flow counters). All three are
     trained on CIC-DDoS2019 day 01-12 and tested on day 03-11 from the raw packet captures
     (`run.bat cic <folder>`, `main.py --models cic`); see `docs/REAL_DATA.md`.

### 3. **Persistent Blacklisting**
   - Automatically saves blocked IPs to `data/blacklist.json`.
   - Restores the blacklist on system restart, ensuring persistent protection against known attackers.
   - Supports manual addition/removal of IPs via the dashboard capabilities.

### 4. **Real-Time Dashboard**
   - **Web Interface**: A modern, dark-themed dashboard built with **Flask** and **Chart.js**.
   - **Live Metrics**: Displays current PPS (Packets Per Second), Drop Rate, Top Attacking IPs, and ML Confidence.
   - **Protocol Analysis**: Visualizes TCP vs UDP vs ICMP traffic distribution.
   - **Attack Alerts**: Shows a history of recent detection events with severity scores.

### 5. **Distributed Attack Simulation**
   - Includes a powerful **Attack Simulator** (`scripts/attack_simulator.py`) capable of generating:
     - UDP/ICMP Floods
     - TCP SYN Floods
     - HTTP Floods
     - **Distributed Botnet Attacks** (Spoofed Source IPs)
   - Supports high-speed packet injection using raw sockets (Scapy L2).

---

## 🏗️ System Architecture

1.  **Ingress Filter (`src/ebpf/xdp_filter.c`)**:
    - The C program compiled into eBPF bytecode.
    - Inspects every incoming packet header.
    - Checks the **Blacklist Map**. If the Source IP is blacklisted -> `XDP_DROP`.
    - Updates **Statistics Maps** (Per-IP packet counts, Protocol counts) -> `XDP_PASS`.

2.  **Userspace Controller (`main.py` -> `src/core/system.py`)**:
    - Loads the XDP program into the kernel.
    - Periodically reads the **Statistics Maps** from the kernel.
    - Feeds traffic data into the **Anomaly Detector**.
    - Updates the dashboard and logs.

3.  **Anomaly Detector (`src/core/anomaly_detector.py`)**:
    - Compares current traffic against the learned **Baseline Profile**.
    - Runs the **ML Classifier** on traffic features.
    - If an attack is detected:
        - Identifies the top offending IPs.
        - Adds them to the **Blacklist Map** (blocking them instantly in the kernel).
        - Triggers an alert.

---

## ⚡ Quick Start (one step)

The launchers create a private Python environment on first use (`.venv-windows` or
`.venv-linux`, so the same folder works from Windows and WSL), install `requirements.txt`,
and reinstall only when that file changes. Python 3.10+ is required.

**Windows**: double-click `run.bat` (dashboard + demo in simulation mode, opens the browser),
or from a terminal in this folder:

```bat
run.bat                 & rem dashboard + demo (press "Guided tour" in the browser)
run.bat test            & rem run the tests
run.bat verify          & rem check every number in the paper/report/slides
run.bat reproduce       & rem rerun every experiment, figure and check (~25 min)
run.bat cic H:/CICDDoS2019  & rem train + test all three levels on the downloaded CIC-DDoS2019 zips
run.bat help            & rem all actions
```

**Linux** (XDP needs BCC first: `sudo apt install python3-bpfcc bpfcc-tools linux-headers-$(uname -r)`):

```bash
bash run.sh test                  # run the tests
bash run.sh dashboard eth0        # protect eth0 with XDP + dashboard (asks for sudo)
bash run.sh live                  # scripted live XDP test on a veth pair (asks for sudo)
bash run.sh check-xdp             # compile + kernel-verifier check (asks for sudo)
bash run.sh check-flows           # kernel counters vs packets with known headers (asks for sudo)
bash run.sh live-cic /mnt/h/CICDDoS2019   # real CIC attack packets through generic XDP (~50 min, sudo)
bash run.sh verify                # check every number in the paper/report/slides
bash run.sh help                  # all actions
```

Both call `run.py`, which can also be run directly: `python run.py <action>`.

---

## 📦 Installation & Requirements

### Prerequisites
- **Linux OS** (Kernel 4.15+ for XDP support).
- **Python 3.10+** (the shipped models need scikit-learn 1.7.2, which requires it).
- **Root privileges** (Sudo) are required to load eBPF programs and send raw packets.

### Install Dependencies
```bash
# System dependencies (for eBPF compilation)
sudo apt update
sudo apt install -y bpfcc-tools linux-headers-$(uname -r) python3-bpfcc clang llvm libelf-dev python3-pip tcpdump

# Python dependencies (from this project/ folder)
sudo pip3 install -r requirements.txt scapy
```

---

## 🛠️ Usage

### 1. Setup Network (Local Testing)
Since XDP filters ingress traffic, testing on a single machine requires a virtual ethernet pair (`veth`).

```bash
# Create veth0 <-> veth1 pair
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 up
sudo ip link set veth1 up
sudo ip addr add 192.168.99.1/24 dev veth0
sudo ip addr add 192.168.99.2/24 dev veth1
```

### 2. Run the DDoS System
Run the system on the "victim" interface (`veth1`).

```bash
sudo python3 main.py --interface veth1 --mode generic --dashboard
```
*   **--interface**: Network interface to protect.
*   **--mode**: `native` (driver support) or `generic` (software emulation, works on all NICs).
*   **--policy**: `p4` (default: source scoring with 30 s blocks, rate limits and a spoofing guard, enforced in XDP) or `top5` (the original rule: permanently block the 5 busiest sources per alert).
*   **--models**: `synthetic` (default, trained on synthetic traffic) or `cic` (trained on real CIC-DDoS2019 captures; also turns on the level-3 flow check, `--no-flow-check` to disable).
*   **--dashboard**: Starts the web UI at `http://127.0.0.1:5000` (add `--host 0.0.0.0` to expose it on the network).

### 3. Simulate an Attack
Run the simulator on the "attacker" interface (`veth0`).

```bash
# Mixed attack from a distributed botnet (100+ random persistent IPs)
sudo python3 scripts/attack_simulator.py --type mixed --distributed --target 192.168.99.2 --duration 30 --interface veth0
```
*   **--distributed**: Spoofs random source IPs (simulating a botnet).
*   **--target**: Must match the IP of the victim interface (`veth1`).
*   **--interface**: Must be the attacker interface (`veth0`).

---

## 📊 Dashboard

The dashboard is accessible at **`http://127.0.0.1:5000`**.

On Windows (no eBPF) the system runs in **simulation mode**: start `python main.py --interface Ethernet --dashboard`, then `python scripts/run_attack_demo.py` to feed it traffic. See [docs/RUN.md](docs/RUN.md).

- **Real-time Status**: Shows current traffic load vs baseline.
- **Top IPs**: Lists the IP addresses sending the most packets.
- **Blacklist**: Shows IPs currently blocked by the XDP filter.
- **ML Analysis**: Displays which features (e.g., "High UDP count") triggered the detection.

---

## 📂 Project Structure

All commands are run from this `project/` folder. Paths are resolved from
`src/config.py`, so `python project/main.py ...` from the repository root works too.

```
project/
├── run.bat / run.sh / run.py  # One-step launchers (Windows / Linux / both)
├── main.py                    # Command-line entry point
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Test configuration
├── src/
│   ├── config.py              # All settings and absolute paths
│   ├── core/                  # Detection pipeline (backend)
│   │   ├── system.py          #   Orchestrator: monitor -> detect -> mitigate -> alert
│   │   ├── traffic_monitor.py #   Loads XDP program, reads eBPF maps, blacklist
│   │   ├── anomaly_detector.py#   Statistical + ML hybrid detection
│   │   ├── traffic_profiler.py#   Long-term traffic profile
│   │   ├── alert_system.py    #   Console/file alerts
│   │   └── metrics_collector.py # CPU/memory metrics
│   ├── dashboard/
│   │   ├── app.py             #   Flask backend + JSON API
│   │   └── static/index.html  #   Single-page dashboard (frontend)
│   ├── ml/                    # Feature extraction, dataset generation, model, trainer,
│   │                          #   cic_data.py (real captures), flow_features.py (level 3)
│   ├── mitigation/            # Mitigation policies + source scorer + evaluation (research)
│   ├── provenance.py          # Code/environment/parameter record stored in every result file
│   └── ebpf/                  # XDP kernel programs (C) + Makefile
├── scripts/
│   ├── reproduce_all.py       # ONE command: every experiment, figure, table, verification, tests
│   ├── run_attack_demo.py     # Feeds simulated attacks to a running system
│   ├── attack_simulator.py    # Sends real packets (Linux, veth testing)
│   ├── benchmark.py           # Control-plane time per monitoring interval
│   ├── mitigation_experiment.py # Mitigation policy comparison + ablations (writes data/results/)
│   ├── safety_experiment.py   # Legitimate traffic that looks like an attack (flash crowds, bulk transfers)
│   ├── mawi_extract.py        # Stream the first minutes of a public MAWI trace (real traffic)
│   ├── trace_experiment.py    # Policy comparison with real legitimate traffic + injected attacks
│   ├── cic_inspect.py         # CIC-DDoS2019: what the CSV labels contain (streams the zips)
│   ├── cic_extract.py         # CIC-DDoS2019: what XDP would see, from the 193 GB of captures
│   ├── cic_experiment.py      # All three levels trained (day 01-12) and tested (day 03-11) on real attacks
│   ├── cic_replay_pcap.py, cic_xdp_replay.sh, cic_xdp_analyze.py  # live replay of real attacks in generic XDP
│   ├── check_xdp_flows.py     # kernel counters vs packets with known headers (Linux, root)
│   ├── verify_results.py      # Checks every number quoted in paper/report/slides -> docs/RESULT_VERIFICATION.md
│   ├── live_xdp_demo.sh       # Live XDP run on a veth pair (Linux, sudo) + live_metrics.json
│   ├── check_xdp_compile.py   # Compile + kernel-verifier check of the XDP program (Linux, sudo)
│   ├── make_figures.py        # Report figures + tables -> docs/figures/
│   └── setup_ebpf.sh / .ps1   # Environment setup (Linux / Windows)
├── tests/                     # pytest suite
├── data/
│   ├── blacklist.json         # Persistent blocked IPs (live XDP mode)
│   ├── traffic_profile.json   # Learned baseline profile
│   └── models/                # Trained model + evaluation reports
├── docs/                      # RUN.md, USAGE_GUIDE.md, PROJECT_EXPLANATION.md,
│                              # REPRODUCIBILITY.md, RESULT_VERIFICATION.md, REAL_DATA.md,
│                              # COMPONENT_STATUS.md, figures/
└── logs/                      # Runtime logs (not in git)
```

Reports, diagrams, research papers and review material are in `../other/`.

---

## ⚠️ Disclaimer
This tool is for **educational and testing purposes only**. Using DDoS tools on networks you do not own is illegal. The XDP filter is powerful and can block all traffic if misconfigured. Always test in a controlled environment.
