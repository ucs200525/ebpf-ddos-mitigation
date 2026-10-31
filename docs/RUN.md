# DDoS Mitigation System - Running the Simulation

All commands below are run from the `project/` folder:
```bash
cd project
```
(Paths are resolved from `src/config.py`, so `python project/main.py ...` from
the repository root also works.)

## 0. Install
```bash
pip install -r requirements.txt
```
On Linux, real XDP also needs BCC: `sudo apt-get install python3-bpfcc`, or run
`sudo ./scripts/setup_ebpf.sh`.

## 1. Start the Mitigation System

**Windows / no eBPF (simulation mode)** - detection, ML, blacklisting and
alerts all run; traffic samples come from the demo script instead of the kernel:
```bash
python main.py --interface Ethernet --dashboard
```

**Linux with eBPF/XDP** (use `generic` if your NIC driver lacks native XDP):
```bash
sudo rm -f data/blacklist.json   # optional: clear blacklist persisted by earlier runs
sudo python3 main.py --interface eth0 --mode generic --dashboard
```
To also accept the demo's injected samples on a live interface, add
`--enable-sim-api` (injected samples can blacklist IPs in the kernel map).

The dashboard binds to `127.0.0.1` by default; add `--host 0.0.0.0` to
expose it on the network.

## 2. Open the Web Dashboard
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## 3. Run the Attack Simulation Demo
In a second terminal (also in `project/`):
```bash
python scripts/run_attack_demo.py
```

This script will:
1. Establish a normal traffic baseline (Phase 1, ~50 pps)
2. Simulate an escalating UDP flood (Phase 2)
3. Simulate a TCP SYN flood (Phase 3)
4. Simulate a coordinated multi-vector DDoS (Phase 4)

Each step is one simulated second; the console shows the score, which detector
fired (`statistical` / `ml` / `hybrid`), the attack type and the blacklist size.

## 4. Verify Mitigation
```bash
curl -s http://127.0.0.1:5000/api/status | python -m json.tool
```

## 5. Tests and model
```bash
python -m pytest                          # full test suite
python -m src.ml.model_trainer train      # retrain the classifier (~2-3 min)
python -m src.ml.model_trainer evaluate   # held-out + end-to-end detector comparison
```
