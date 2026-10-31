# Reproducibility and evidence

This file documents how every result was produced and what each kind of result can and
cannot show. Each result file in `data/` carries a `provenance` block (git commit, whether
`project/` had uncommitted changes, creation time, Python/NumPy/scikit-learn versions,
platform, parameters and SHA-256 of the input models). `docs/RESULT_VERIFICATION.md` maps
every number quoted in the paper, report and slides to its result file. The provenance block
also stores `code_sha256`, one hash over every file in `src/` and `scripts/`, so a result can
be tied to the exact code even when that code was not yet committed.

```bash
cd project
pip install -r requirements.txt
python scripts/reproduce_all.py               # all experiments, figures, verification, tests
python scripts/reproduce_all.py --check-model # also retrain a copy of the detection model and compare
```

`reproduce_all.py` writes `data/results/reproduce_manifest.json` (command, exit code, duration
and output hashes of every step). The live XDP run and the kernel-verifier check need Linux
and root and are not part of it (see the last section).

## 1. Levels of evidence

The project contains five different kinds of evidence. They must not be mixed up.

| Level | What it is | What it shows | What it does not show |
|---|---|---|---|
| Implemented | Code exists and unit tests pass | The logic does what the tests check | Anything about real traffic |
| Synthetic simulation | Labelled runs from `src/ml/traffic_dataset.py`, replayed second by second through the real detector and policy code | Behaviour on traffic shaped by our assumptions, with exact ground truth per source | Accuracy on real attacks; the generator encodes our assumptions |
| Trace-driven | Real legitimate traffic (MAWI backbone trace, 180 s) with **injected** synthetic attacks | How policies treat real clients (real rates, churn, headers) | Behaviour against real, captured attacks; malicious background in the trace is counted as legitimate |
| Real captured attacks | CIC-DDoS2019 packet captures (`docs/REAL_DATA.md`): train on day 01-12, test on day 03-11, all three levels, every packet real | Detection, per-flow classification and collateral damage on real attacks and real benign traffic, with the dataset's own labels | Separation of many attack sources (every attack comes from one address); behaviour on networks with much heavier benign traffic |
| Live XDP, real attacks | Real CIC-DDoS2019 test-day packets replayed with tcpreplay into the running system, generic XDP on veth pairs in WSL2 (`run.sh live-cic`, `docs/REAL_DATA.md`) | Drops measured in the kernel on real attack and benign packets; the implementation works end to end | Throughput on a physical NIC or in native mode; statistics over many runs |
| Live XDP | The full system on a veth pair in WSL2, generic XDP, one attacker in its own network namespace | The kernel side works end to end (maps, expiry, rate limits, spoofing guard) | Throughput or overhead on a physical NIC in native mode; single runs, not statistics |

The shipped detection model's accuracy (0.995 on held-out windows) is a **synthetic** result:
that model was trained and tested on synthetic traffic only. The real-attack tests (CIC-DDoS2019,
`scripts/cic_experiment.py`, results in `data/results/cic_results.json` and at the end of
`docs/figures/RESULTS_TABLES.md`) measure that shipped model on real attacks too, and train a
second model set (`--models cic`) on the real captures.

## 2. Environment

| Item | Value |
|---|---|
| Experiments (synthetic, trace, safety, benchmark) | Windows 11, Python 3.10.6, NumPy 1.26.4, scikit-learn 1.7.2, 16 logical CPUs, 32 GB RAM |
| Live XDP runs | Kali Linux (WSL2), kernel 6.6.87.2-microsoft-standard-WSL2, Python 3.13, BCC from the distribution |
| XDP mode | Generic (SKB) mode on a veth pair; native mode was not tested |
| Interface | `veth1` (victim) in the default namespace, `veth0` (attacker) in namespace `ddos_attacker` |

The exact versions of each run are in its provenance block. scikit-learn is pinned in
`requirements.txt` because pickled models only load reliably in the version that saved them.

## 3. Detection model

| Item | Value |
|---|---|
| Algorithm | Random Forest (scikit-learn), `class_weight='balanced'`, `random_state=42` |
| Features | 16 window-level features over the last 10 s (`src/ml/feature_extractor.py`, `FEATURE_SET_VERSION = 2`) |
| Training data | 600 synthetic runs, seed 7 (55,256 windows); 11 scenarios, 60% attack runs |
| Test data | 300 independent synthetic runs, seed 1007 (27,262 windows) |
| Model selection | Grid over trees {100, 200}, depth {8, 12, none}, min leaf {2, 5}; 5-fold `GroupKFold` by run; simplest model within one std of the best CV F1 |
| Chosen | 100 trees, no depth limit, min leaf 2 |
| Second evaluation | `evaluate --seed 3030 --runs 300 --e2e-runs 200`: 300 fresh runs for window metrics (27,389 windows) and 200 runs (seed 3031) replayed second by second through the rules-only and hybrid detectors |

Two different window totals appear in the documents because there are two evaluations:
27,262 windows (the test set stored with the model) and 27,389 windows (the separate
`evaluate` run). Both are synthetic.

## 4. Mitigation policies and parameters

| Parameter | Value | Where |
|---|---|---|
| P4 thresholds | `t_block` and `t_limit` chosen on tuning seed 901 by maximising attack dropped − 2 × legitimate dropped; values in `mitigation_results.json` → `protocol.chosen_params` | `scripts/mitigation_experiment.py` |
| P4 scorer | Logistic regression on 5 features (known, rate jump, relative rate, SYN share, age), trained on seed 900, at most 50 sources per class per anomalous second | `src/mitigation/source_scorer.py`, `evaluation.collect_source_samples` |
| Block / throttle expiry | 30 s | `SourceScoringPolicy(ttl=30)` |
| Throttle rate | 99th percentile of normal per-source rates, at least 10 packets/s | `SourceHistory.per_source_limit` |
| Known source | Seen in at least 2 intervals judged normal | `SourceHistory.known` |
| Spoofing fallback | On when ≥ 100 brand-new sources carry ≥ 50% of the interval's traffic; unknown sources then share what is left of 1.5 × baseline after known sources | `SourceScoringPolicy` |
| Attack mode hold | 5 s after the last anomalous second | `MitigationPolicy(hold=5)` |
| P0 / P1 | Top 5 sources with > 15 packets in the interval; permanent / 30 s | `TopKBlacklist` |
| P2 | During attack mode, drop sources not known from normal intervals (Peng et al. 2003) | `HistoryFilter` |
| P3a / P3b | Cap total admitted traffic at 1.5 × baseline / cap each source at the p99 normal rate | `AggregateRateLimit`, `PerSourceRateLimit` |
| AHIF | Admit sources by their normal-time traffic share until 1.5 × baseline (Goldstein et al. 2008), /32 (and /24 on the trace) | `AdaptiveHistoryFilter` |
| SIG / SIG* | Drop header signatures (protocol, TTL, TCP window or size) that a Random Forest trained on seed 900 labels malicious (Dimolianis et al. 2021); SIG* uses ground-truth labels (upper bound) | `SignatureFilter` |

**Baselines are our re-implementations** from the published descriptions. We did not have
the original authors' code, and the original systems differ in details we could not
reproduce (for example, AHIF's prefix aggregation and SIG's exact feature set and data plane).

**Fairness.** Every policy is evaluated on the same runs, the same cached per-second
detector verdicts (hybrid detector unless stated) and the same per-source counters. The
only extra input any policy gets is the per-source header signature used by SIG.

## 5. Evaluation protocol

| Experiment | Runs | Seeds | Aggregation |
|---|---|---|---|
| Synthetic mitigation | 100 runs per seed, 3 realism settings | test 2001–2005 (tuning 900, 901) | mean ± std over the 5 seeds |
| Trace-driven | 20 destinations × 2 windows of 90 s = 40 runs per seed | injection 7000–7004 | mean ± std over the 5 seeds |
| Safety scenarios | 40 runs per scenario per seed, realistic setting | 5001–5005 | mean ± std over the 5 seeds |
| Control-plane benchmark | 20 synthetic runs (realistic), every interval timed | 6001 | mean, p50, p99, max over intervals |

Metrics (`src/mitigation/evaluation.py`): attack dropped and passed, legitimate dropped and
retained (all in packets, aggregated over every run of a seed), legitimate loss split into
attack runs and attack-free runs, legitimate sources hit, attack runs mitigated (≥ 50% of
attack packets dropped in some second) and time to mitigate, attack runs detected and
detection delay, legitimate loss after the attack ends and time to recover (< 1% loss in a
second).

The model of the data path: a decision made in second *t* affects packets from second
*t + 1*; XDP counts every packet before dropping it, so the detector sees the full load.

**Design history.** A first version of P4 (results archived in
`data/results/mitigation_results_v1.json`) blocked legitimate users of small networks. Its
features, training sampling and spoofing trigger were changed, the test seeds it had been
inspected on (1001–1005) were retired, and all reported synthetic results use the unseen
seeds 2001–2005.

## 6. Live XDP measurements (Linux, root)

```bash
cd project
sudo python3 scripts/check_xdp_compile.py      # compile + kernel verifier
sudo bash scripts/live_xdp_demo.sh             # writes logs/live_demo_<time>/live_metrics.json
sudo POLICY=top5 bash scripts/live_xdp_demo.sh
```

`live_metrics.json` records the XDP program's cost per packet from the kernel's BPF
statistics (`run_time_ns / run_cnt`), the control plane's CPU and memory, detection and
mitigation delay, and packets sent versus dropped. These are measurements of generic XDP
on a virtual link and must be reported as such, not as NIC throughput.
