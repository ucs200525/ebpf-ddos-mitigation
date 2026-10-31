# Component status (31 October 2026)

What exists, what was run, and on what evidence. Compared with the reference architecture of
Dimolianis et al. (IEEE Access 2021, Fig. 1: signature extraction, classification, reduction,
XDP mitigation, traffic redirection at an edge router), which is cited in the paper as [7] and
re-implemented here only as the SIG baseline.

## Our system

| Component | Implemented in | Tested how | Status |
|---|---|---|---|
| L3/L4 packet parsing (Ethernet, 802.1Q, IPv4, TCP/UDP ports and flags; no ports in later fragments) | `src/ebpf/xdp_filter.c` (BCC), `xdp_filter_libbpf.c` | Kernel verifier as root (`scripts/check_xdp_compile.py`: OK); 26 kernel checks with known packets (`scripts/check_xdp_flows.py`: all OK); the capture parser used for training matches dpkt on 1.17 M packets | Done, verified in the kernel |
| Counters: totals, per source (SYN without ACK), per flow (13 counters, LRU 262k), all in IP bytes | XDP maps; `src/core/traffic_monitor.py` (batched lookup-and-delete of the flow map) | `check_xdp_flows.py` (every counter equals what was sent) | Done, verified in the kernel |
| Enforcement: block with expiry, token-bucket rate limit, spoofing guard, SYN guard (per source, per second) -> XDP_DROP / XDP_PASS | XDP program + maps | Unit tests; live runs (drops counted by tcpdump after XDP match XDP's own counters exactly) | Done, verified live |
| Level 1: attack detection (rules + Random Forest, 16 window features) | `src/core/anomaly_detector.py`, `src/ml/` | Synthetic held-out runs; CIC-DDoS2019 test day (98.7% of attack windows, shipped model); live: attack drop reached 50% within 2 s in every real attack run | Done |
| Level 2: source scoring (P4) | `src/mitigation/policies.py`, `live.py` | Synthetic seeds, MAWI trace, CIC test day, live XDP | Done |
| Level 3: flow classification, P4+L3 | `src/ml/flow_features.py`, `FlowVerifiedScoring`, `LiveFlowEvidence` | CIC test day (99.9% of attack flows, 0.21% of benign flows); live XDP with the block-time check (legitimate loss during attacks 6.9% -> 0.4%, in attack-free stretches 51.3% -> 0.003%, attack drop 99.8%) | Done; the fix came from a test-day failure, so not a held-out result; attackers imitating normal flows are untested |
| Training on real attacks | `scripts/cic_inspect.py`, `cic_extract.py`, `cic_experiment.py`, `src/ml/cic_data.py` | Train day 01-12, test day 03-11; results in `data/results/cic_results.json` | Done |
| Live replay of real attacks in generic XDP | `scripts/cic_replay_pcap.py`, `cic_xdp_replay.sh`, `cic_xdp_analyze.py`, `xdp_counters.py` | Kali Linux (WSL2, kernel 6.6.87) as root; `data/results/cic_xdp_results.json` | Done |
| Dashboard, alerts, Windows demonstration mode | `src/dashboard/`, `src/core/alert_system.py` | Unit tests, manual demo | Done (no result comes from the demonstration mode) |

## Components of the reference architecture

| Reference component | Here | Status |
|---|---|---|
| Signature extraction (XDP-based, from mirrored traffic) | Per-source header signature (TTL, TCP window or size) is taken from the datasets offline (`scripts/cic_extract.py`, trace experiment); the live XDP program does not export signatures | Offline only (SIG baseline) |
| Signature classification | Random Forest over signatures (`train_signature_classifier`, SIG baseline) | Offline only |
| Signature reduction (merging signatures into fewer mitigation rules) | Not implemented | Missing |
| Anomaly mitigation in XDP | Blocks, rate limits, spoofing guard and SYN guard in the XDP program (decided per source, not per signature) | Done |
| Traffic mirroring and redirection at an edge router, separate protection service | Not implemented: our system runs on the protected host's (or gateway's) own interface | Different design; not implemented |
| Benign traffic forwarded, attack dropped | XDP_PASS / XDP_DROP on the same host | Done |

## Environment limits (WSL2)

- Only generic XDP on veth pairs was run. Native XDP needs a physical NIC whose driver supports it; WSL2's
  virtual NIC does not, so throughput and per-packet cost on real hardware are not measured
  (generic-mode cost on veth: about 1.4-1.6 microseconds per packet, from bpf_stats).
- BCC needs the kernel headers; in WSL2 they come from the `kheaders` module, which only root can load.
- Replayed frames were cut to 128 bytes (headers only); XDP's byte counters use the IP total length, which is
  unchanged, so features are not affected. Replay rate was up to about 27,000 attack packets/s; at most 8
  packets per run were lost before XDP.
