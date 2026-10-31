# Real attack traffic: CIC-DDoS2019 for all three levels

This is how the three levels are trained and tested on **real packet captures**
instead of synthetic traffic. Everything is computed from the packets, with the
same feature code the running system uses, so a model trained here can be
loaded by `main.py --models cic`.

| Level | Question it answers | Unit | Model | Live input |
|---|---|---|---|---|
| 1 | Is the network under attack now? | 10-s window, updated every second | random forest, 16 window features | XDP totals + per-source counters |
| 2 | Which sources are attacking? | source x second | logistic regression, 5 counter features (P4 scorer) | XDP per-source counters + history |
| 3 | Does this source's traffic look like an attack? | inbound flow (5-tuple) x 10 s | random forest, 16 flow features | XDP flow map (counters added for this) |

Level 3 is used by the P4+L3 policy (`FlowVerifiedScoring`). A source whose
recent flows look legitimate is never blocked, at most slowed down. A source
whose flows look like an attack is blocked once its level-2 score reaches the
lower threshold.

## Steps

The dataset (Sharafaldin et al., ICCST 2019) is downloaded from
<https://www.unb.ca/cic/datasets/ddos-2019.html>: `CSVs/CSV-01-12.zip`,
`CSVs/CSV-03-11.zip` and the five `PCAPs/...` zips (about 30 GB). Keep the
folder layout (`CSVs/`, `PCAPs/01-12/`, `PCAPs/03-11/`). Nothing is unzipped:
the scripts stream the files out of the zips.

```bash
run.bat cic H:/CICDDoS2019        # Windows (or: bash run.sh cic /path/to/CICDDoS2019)
```

That runs three steps, which can also be run alone from `project/`:

1. `python scripts/cic_inspect.py --cic-dir DIR` reads every CSV (31 GB) and
   records, per file and label, the flow count, time range, protocols and the
   addresses involved. Writes `data/results/cic_inventory.json`.
2. `python scripts/cic_extract.py --cic-dir DIR` parses every capture (193 GB,
   965 files, about 10 min on 8 cores). It keeps only the IPv4 packets that
   enter the victim network 192.168.50.0/24 from outside it, which is what an
   XDP filter at the network's ingress sees, and writes two tables per day to
   `data/external/cic/<day>.npz` (not in git):
   - `persec`: per (second, source): packets, bytes, SYN without ACK, TCP/UDP/ICMP
     counts, and the header signature of the first packet;
   - `flows`: per (10-s window, 5-tuple): the flow counters of level 3.
     Flows from the attack address are hash-sampled 1 in 32 because the floods
     create tens of millions of them; every other flow is kept. The hash is
     taken over the flow key, so a flow that crosses two capture files is kept
     in both or in neither, and the parts are merged afterwards.
   The header parser was checked against `dpkt` on 1.17 million packets
   (all fields identical).
3. `python scripts/cic_experiment.py` trains and tests the three levels.
   It writes `data/results/cic_results.json` and
   `data/models/cic_{window_model,flow_model,source_scorer}.joblib`.
   `--stages level1|level3|level2` reruns a single stage.

`scripts/reproduce_all.py` includes step 3 when the extracted files exist.

## Labels

What the CSV labels of the dataset actually mean was checked first (`cic_inventory.json`):

- Every flow labelled as an attack has source 172.16.0.5, the testbed's attack
  machine. The few other addresses in attack labels are the victim's replies
  (192.168.50.1 on 01-12, 192.168.50.4 on 03-11).
- The attack labels cover each day without gaps. For example, "Portmap" covers
  09:18-10:01 local time, although the published attack ran 09:43-09:51.
  The label means "traffic from the attack machine at that time".
- The CSV timestamps are local time (Atlantic: UTC-3 on 03-11, UTC-4 on 01-12).
  On both days the first labelled flow starts within 1.2 s of the first
  captured packet, which confirms these offsets.

So the ground truth here is:

| Item | Attack when |
|---|---|
| packet, flow, source entry | source address is 172.16.0.5 (the dataset's own labelling) |
| attack type | the CSV label whose time range contains it |
| second (level 1) | the attack machine sent >= 100 packets in that second |

The attack machine also sends a little background traffic between its attacks
(median 4 packets/s on 01-12, 0 on 03-11). CIC labels it as attack. Levels 2 and 3 keep that label (it is the
dataset's), and level 3 also reports recall inside and outside the attack
episodes. For level 1 such a second is not "under attack", because there is
no load to detect.

## Train/test split

Day **01-12** (the dataset's training day) is used for every fit and every
threshold: cross-validation on 10-minute blocks, the scorer, and P4's
thresholds. Day **03-11** (its testing day) is used only for testing. Portmap
appears only on 03-11, so it is an attack type that was never seen in training.

## What the data allows and what it does not

- **One attack address.** Every attack packet comes from 172.16.0.5. Blocking
  the single heaviest source stops every attack in this dataset. So this test
  measures detection, flow classification and collateral damage on real
  benign traffic. It cannot measure how well a policy separates *many* attack
  sources; the synthetic and MAWI-injection experiments are still needed for that.
- **Light benign traffic.** 13-18 packets/s on average (bursts up to about
  8,800/s), from 587 (01-12) and 1,268 (03-11) distinct external sources.
  The floods run at a median of 16,000-27,000 packets/s per episode on 03-11 and
  up to 42,000 on 01-12 (one 01-12 UDP-lag episode runs at 84). A detector
  trained only on this learns "thousands of packets/s = attack", so its false
  alarms and its misses on *other* networks are measured separately
  (`level1.cross_dataset`: MAWI benign traffic, MAWI with injected attacks,
  synthetic test runs).
- **Two attacks never show up at the ingress.** The 03-11 Portmap and UDP-lag
  attacks never reach flood rates there: outside the flood episodes the attack
  machine averages 1.8 and 4.3 packets/s during their label periods, at most 432
  and 104 in any second. (The label periods also cover the first seconds of the
  next flood.) They create no load, so nothing should detect or mitigate them
  as floods.
- **Reflection is replayed from one host.** The "DrDoS" attacks arrive with
  the reflector's service port as source port (53, 123, 389, ...) but all from
  172.16.0.5. Real reflection comes from thousands of reflectors.
- **Fields that identify the machine were excluded.** Level 3 never sees
  addresses, raw port numbers, TTL or TCP window. In a testbed with one attack
  machine, these identify the machine rather than the attack.

## Using the models

```bash
python main.py --interface Ethernet --dashboard --models cic          # Windows, simulation
sudo python3 main.py --interface eth0 --mode generic --dashboard --models cic   # Linux, XDP
```

`--models cic` loads:

- the window detector trained on synthetic traffic **plus** CIC 01-12;
- the flow classifier trained on CIC 01-12 flows (P4+L3, which releases sources whose
  flows look legitimate; that mode was chosen on 01-12);
- the **synthetic-trained** scorer and thresholds. The scorer trained on CIC learned
  its single attack machine and does not transfer (see Findings).

`--no-flow-check` turns level 3 off. In simulation mode there is no flow map, so level 3 has no evidence and
P4+L3 behaves exactly like P4.

The XDP program keeps the level-3 counters per flow: packets, IP-length sum,
min, max and sum of squares, first and last time, sum of squared and largest
inter-arrival time, and SYN/ACK/FIN/RST/PSH counts. User space reads and
clears the map every 10 s (`TrafficMonitor.get_flow_records`). The
per-source SYN counter now counts SYN-without-ACK, the definition used in
training. Later IP fragments get no ports, as offline.

## Findings

All numbers are from `data/results/cic_results.json`. The full tables are at the
end of `docs/figures/RESULTS_TABLES.md`, generated from that file.

**Level 1 (is there an attack).**
- On the real test day, every window model finds the attacks:
  - shipped synthetic-trained model: 98.7% recall, 1.1% false alarms;
  - CIC-only model: 99.2% recall, 0.08% false alarms;
  - combined model: 99.3% recall, 0.05% false alarms.
- So the model trained only on synthetic traffic does transfer to these real attacks.
- The CIC-only model does **not** transfer the other way: it catches 15% of the
  attacks injected into MAWI traffic and 35% of the synthetic test attacks. It
  learned the testbed's rates (thousands of packets/s). That is why the deployed
  set uses the combined model (90.5% and 98.5% there).
- On MAWI benign traffic the window models alone (0.5 cut) flag 27% (shipped) and
  35% (combined) of windows. This is the same weakness the trace experiment showed
  (9.1% false-alarm seconds for the deployed hybrid detector).
- As deployed (hybrid detector, second by second, the whole test day): all 6
  attack episodes detected in their first second, about 0.4-0.5% of benign
  seconds flagged. The statistical rules alone flag 11.2%.

**Level 3 (does this traffic look like an attack).**
- On the test day the flow classifier catches 99.9% of attack flows and flags
  0.21% of benign flows (0.15% of benign packets). That includes Portmap, which
  never occurs in training.
- The benign flows it gets wrong are mostly real DNS answers from public resolvers
  (4.2.2.4, 8.8.8.8). They look exactly like reflected DNS.
- The hand rules: "SYN-only TCP flow = attack" catches the SYN flood (99.9%) and
  nothing else (28.2% overall), with no false alarms. "ACK-only TCP flow = attack"
  catches nothing and flags 55.8% of benign flows, because an established TCP
  connection looks "ACK-only" inside a 10-second window.
- These scores are this dataset's: one attack machine and replayed reflection make
  flows easy to tell apart. Expect lower numbers on other networks.

**Level 2 + mitigation (which sources; collateral damage).** Test day, all policies:

| Policy | Attack dropped | Legitimate dropped |
|---|---|---|
| P0 (top-5 blacklist) | 99.9% | 24.0% |
| P4 | 99.8% | 14.4% |
| P4+L3 | 99.8% | 10.9% |

- Nearly all of P4's loss comes from blocking benign sources during their heavy
  bursts (downloads; one external server, 8.250.89.254, accounts for the largest
  share). In a diagnostic replay, about two-thirds of the lost packets were lost
  before the first 10-s flow window could vouch for the source. P4+L3 releases the
  block once it does; what remains is that first window.
- P2 (history filter) drops **nothing**: the attack machine sent background
  traffic before attacking, so it counted as a known source.
- SIG with its synthetic-trained classifier drops 39.5%.
- A scorer trained on CIC 01-12 reaches 0.17% loss here, but it learned the single
  attack machine. On synthetic traffic it stops 42.7% of attacks and drops 20.5% of
  legitimate traffic, against 80.8% and 1.4% for the shipped scorer. It is
  reported, not deployed.

**What this changes for the claims.** Real attacks confirm level 1 and level 3 on
this dataset. They also confirm the collateral-damage problem of P0 and P4 on real
benign traffic, and that level 3 reduces it without losing attack coverage. They
do not test many-source attacks; every attack here comes from one address.

## Live: the real packets through the running system (generic XDP)

The offline results model the kernel's drops in Python. `run.sh live-cic <folder>` (Linux, root) measures them in
the kernel. Run on 31 Oct 2026 in Kali Linux (WSL2, kernel 6.6.87), with the final code:

1. `scripts/cic_replay_pcap.py` copies the original frames of five test-day attack episodes (2 min before, the
   first 3 min of the attack, 1 min after) and the two attack-free 5-minute stretches with the most benign packets
   into pcap files, cut to 128 bytes (headers only).
2. `scripts/cic_xdp_replay.sh` replays each file with tcpreplay at its original timing into four veth pairs at
   once: top-5, P4, P4 with the synthetic + CIC detector, P4+L3. `main.py --mode generic` runs on the far end of
   each, started fresh for every run. tcpdump on the victim end records only what XDP passed; bpftool reads
   XDP's own counters.
3. `scripts/cic_xdp_analyze.py` compares sent and passed packets per source and replays the same windows offline.
   Results: `data/results/cic_xdp_results.json`.

| Configuration | Attack dropped | Legit dropped, attack runs | Legit dropped, attack-free runs |
|---|---|---|---|
| top-5 | 99.9% | 31.1% | 51.6% |
| P4 | 99.8% | 6.9% | 51.3% |
| P4, synthetic + CIC detector | 99.8% | 7.3% | 49.0% |
| P4+L3 | 99.8% | 0.4% | 0.003% |

XDP's pass counts equal tcpdump's in all 28 run-configuration pairs, tcpdump dropped nothing, and at most 8
packets per run were lost before XDP.

**Block-time check.** The attack-free losses come from legitimate downloads that arrive in 2-3 s bursts of
thousands of packets per second: they raise an alarm and are blocked within a second. The first version of this
experiment (29 Oct) gave P4+L3 evidence only from complete 10 s windows; the verdict came after the burst, and it
lost 4.2% during the attacks and 43.3% in the attack-free stretches. P4+L3 now also reads a source's flows so far,
without clearing the map, just before acting on it. **This failure was found on the test day, so the improved
numbers are not a held-out result.** No threshold changed and no model was retrained.

**Does the veto shield attacks?** Our own flood tool is unlike anything in the training data (2,000 packets/s from
100 forged addresses): P4+L3 dropped 89.0% of a UDP flood (P4: 52.8%) and 76.8% of a SYN flood (P4: 75.5%). An HTTP run
gave no result, because the tool cannot hold TCP connections with forged addresses. An attacker who deliberately
imitates normal flows has not been tested; the veto would spare it.

**Not checked yet.** The training-day check that the classifier, trained on complete windows, also judges partial
flows well needs the per-second flow table of 01-12, and the offline figure for the whole test day (P4+L3 10.9%,
window evidence only) needs that table for 03-11. Both come from `scripts/cic_extract.py`, which needs the
downloaded zips; the drive was disconnected on 31 Oct. Until then the table exists only for the replay windows
(`scripts/cic_replay_fsec.py`), and `scripts/cic_experiment.py --stages level2` stops with a message asking for it.
