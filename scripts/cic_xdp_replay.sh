#!/usr/bin/env bash
# Live experiment: real CIC-DDoS2019 traffic through the real system, generic XDP (Linux, root).
#
# For every replay file made by scripts/cic_replay_pcap.py, four configurations run side by
# side, each on its own veth pair (rpaN -> rpbN) with its own state folder:
#   P0     --policy top5 (the original rule: block the 5 busiest sources per alert, permanently)
#   P4     --policy p4 --models synthetic --no-flow-check
#   P4cic  --policy p4 --models cic --no-flow-check   (detector trained on synthetic + CIC 01-12)
#   P4L3   --policy p4 --models cic                   (+ level 3: flow classifier verifies P4)
# Each one: tcpdump on rpbN (generic XDP runs before packet taps, so it records only the
# packets XDP let through), main.py --mode generic on rpbN, then tcpreplay of the run at
# its original timing on rpaN. After the replay XDP's own counters are read (bpftool).
# Every system instance starts fresh for every run, as in the offline experiment.
#
#     sudo bash scripts/cic_xdp_replay.sh /root/cic_replay [/root/cic_xdp_runs/<name>]
# Then: python scripts/cic_xdp_analyze.py <that output folder>

set -uo pipefail
REPLAY=${1:-/root/cic_replay}
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
OUT=${2:-/root/cic_xdp_runs/$(date +%Y%m%d_%H%M%S)}
PY="$PROJECT_DIR/.venv-live/bin/python"
[ "$(id -u)" -eq 0 ] || { echo "run as root"; exit 1; }
[ -x "$PY" ] || { echo "missing $PY (run scripts/live_xdp_demo.sh once to create it)"; exit 1; }
for t in tcpreplay tcpdump bpftool; do command -v $t >/dev/null || { echo "missing $t"; exit 1; }; done

NAMES=(P0 P4 P4cic P4L3)
ARGS=("--policy top5 --models synthetic --no-flow-check"
      "--policy p4 --models synthetic --no-flow-check"
      "--policy p4 --models cic --no-flow-check"
      "--policy p4 --models cic")
mkdir -p "$OUT"
cp "$REPLAY/manifest.json" "$OUT/"
{ echo "kernel: $(uname -r)"; echo "date: $(date -Is)"; tcpreplay --version 2>&1 | head -1;
  tcpdump --version 2>&1 | head -1; bpftool version | head -1; nproc; } > "$OUT/environment.txt"

cleanup() {
    for i in "${!NAMES[@]}"; do ip link del "rpa$i" 2>/dev/null || true; done
    pkill -f "main.py --interface rpb" 2>/dev/null || true
}
trap cleanup EXIT

# Keep the headers module loaded: BCC otherwise loads and unloads it on every start, and parallel
# starts cut off each other's read. One compile first, so the headers are unpacked once.
modprobe kheaders 2>/dev/null || true
"$PY" scripts/check_xdp_compile.py 2>&1 | tail -1 | tee -a "$OUT/environment.txt"

RUNS=$("$PY" -c "import json,sys; print(' '.join(r['name'] for r in json.load(open(sys.argv[1]))['runs']))" "$REPLAY/manifest.json")
for run in $RUNS; do
    echo "==> run $run"
    declare -a MAIN TCPD REPLAY_PIDS
    for i in "${!NAMES[@]}"; do
        D="$OUT/$run/${NAMES[$i]}"; mkdir -p "$D/state"
        ip link del "rpa$i" 2>/dev/null || true
        ip link add "rpa$i" type veth peer name "rpb$i"
        for dev in "rpa$i" "rpb$i"; do
            sysctl -qw "net.ipv6.conf.$dev.disable_ipv6=1"
            ip link set "$dev" up
        done
        tcpdump -i "rpb$i" -s 96 -B 262144 -n -w "$D/passed.pcap" ip 2> "$D/tcpdump.log" &
        TCPD[$i]=$!
        DDOS_STATE_DIR="$D/state" "$PY" main.py --interface "rpb$i" --mode generic ${ARGS[$i]} \
            > "$D/console.log" 2>&1 &
        MAIN[$i]=$!
    done
    for i in "${!NAMES[@]}"; do
        D="$OUT/$run/${NAMES[$i]}"
        for _ in $(seq 1 120); do grep -q "XDP program loaded successfully" "$D/console.log" && break; sleep 1; done
        grep -q "XDP program loaded successfully" "$D/console.log" || { echo "${NAMES[$i]}: XDP did not load"; tail "$D/console.log"; exit 1; }
        ip -d link show "rpb$i" | grep -o "xdpgeneric[^ ]*" > "$D/xdp_mode.txt" || true
    done
    sleep 3
    for i in "${!NAMES[@]}"; do
        D="$OUT/$run/${NAMES[$i]}"
        tcpreplay -i "rpa$i" "$REPLAY/$run.pcap" > "$D/tcpreplay.log" 2>&1 &
        REPLAY_PIDS[$i]=$!
    done
    for i in "${!NAMES[@]}"; do wait "${REPLAY_PIDS[$i]}"; done
    sleep 3
    for i in "${!NAMES[@]}"; do
        D="$OUT/$run/${NAMES[$i]}"
        "$PY" scripts/xdp_counters.py "rpb$i" > "$D/xdp_counters.json" 2>&1
        kill -INT "${MAIN[$i]}" 2>/dev/null
    done
    sleep 4
    for i in "${!NAMES[@]}"; do
        kill -INT "${TCPD[$i]}" 2>/dev/null
        kill -9 "${MAIN[$i]}" 2>/dev/null || true
    done
    wait 2>/dev/null
    for i in "${!NAMES[@]}"; do ip link del "rpa$i" 2>/dev/null || true; done
    for i in "${!NAMES[@]}"; do
        D="$OUT/$run/${NAMES[$i]}"
        echo "    ${NAMES[$i]}: $(grep -o 'Actual: [0-9]* packets[^)]*)' "$D/tcpreplay.log" | head -1) | " \
             "tcpdump $(grep -o '[0-9]* packets captured' "$D/tcpdump.log") $(grep -o '[0-9]* packets dropped by kernel' "$D/tcpdump.log")"
    done
done
echo "done -> $OUT"
