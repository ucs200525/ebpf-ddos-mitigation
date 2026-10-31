#!/usr/bin/env bash
# Live XDP demo on one Linux machine (also works in WSL2 Kali/Ubuntu).
#
# Creates a veth pair (veth0 = attacker, in its own network namespace "ddos_attacker";
# veth1 = victim), starts the system on
# veth1 in generic XDP mode with the dashboard, sends light normal traffic so
# a baseline is learned, then a UDP flood from 100 forged bot addresses, and saves the
# evidence (console log, alerts, final status JSON) under logs/live_demo_<time>/.
#
# Run it yourself (it needs root):
#     cd project
#     sudo bash scripts/live_xdp_demo.sh
# Open http://127.0.0.1:5055 in a browser while it runs to watch the dashboard (DASH_PORT=5055;
# a separate port, because WSL2 shares ports with Windows, where the demo dashboard may use 5000).
#
# Measurements written to logs/live_demo_<time>/live_metrics.json:
#   - XDP program cost per packet (kernel bpf_stats: run_time_ns / run_cnt; generic XDP on veth,
#     so it is NOT a NIC/native-mode throughput number)
#   - control-plane CPU % and RSS of main.py (sampled once per second from /proc)
#   - detection delay (attack start -> first anomalous interval) and mitigation delay
#     (attack start -> first interval with >= 50% of incoming packets dropped)
#   - packets sent by the attacker vs packets dropped by XDP
#
# Options (environment variables):
#     BASELINE_SECONDS=40   ATTACK_SECONDS=30   ATTACK_PPS=2000   KEEP_RUNNING=0
#     POLICY=p4 (source scoring, default) or top5     ATTACK_TYPE=udp or tcp (SYN flood)
#     MODELS=synthetic (default) or cic (detector trained on synthetic + CIC, P4+L3 flow check)
#     e.g.  sudo POLICY=top5 bash scripts/live_xdp_demo.sh

set -euo pipefail

BASELINE_SECONDS=${BASELINE_SECONDS:-40}
POLICY=${POLICY:-p4}
export MODELS=${MODELS:-synthetic}
ATTACK_TYPE=${ATTACK_TYPE:-udp}
DASH_PORT=${DASH_PORT:-5055}
ATTACK_SECONDS=${ATTACK_SECONDS:-30}
ATTACK_PPS=${ATTACK_PPS:-2000}
KEEP_RUNNING=${KEEP_RUNNING:-0}

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
OUT="logs/live_demo_$(date +%Y%m%d_%H%M%S)"

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run with sudo:  sudo bash scripts/live_xdp_demo.sh"
    exit 1
fi

step() { echo; echo "==> $*"; }

step "Checking dependencies"
missing=""
python3 -c "import bcc" 2>/dev/null || missing="$missing python3-bpfcc"
python3 -c "import scapy" 2>/dev/null || missing="$missing python3-scapy"
command -v ip >/dev/null || missing="$missing iproute2"
if [ -n "$missing" ]; then
    echo "Installing:$missing"
    apt-get update -qq && apt-get install -y -qq $missing
fi

# Python packages for the project, in a venv that can also see the system bcc module
if [ ! -x .venv-live/bin/python ]; then
    step "Creating Python environment (.venv-live, first run only)"
    python3 -m venv --system-site-packages .venv-live
fi
# Re-sync every run: the shipped models need the pinned scikit-learn version
.venv-live/bin/pip install -q -r requirements.txt
PY="$PROJECT_DIR/.venv-live/bin/python"

# The attacker end lives in its own network namespace. With both ends in the same
# namespace the kernel treats 192.168.99.2 as a local address and the traffic does
# not cross the veth cable, so XDP on veth1 would never see it.
NS=ddos_attacker
ATT="ip netns exec $NS"

step "Creating veth pair: veth0 (attacker, 192.168.99.1, namespace $NS) <-> veth1 (victim, 192.168.99.2)"
ip link del veth1 2>/dev/null || true
ip netns del "$NS" 2>/dev/null || true
ip netns add "$NS"
ip link add veth0 type veth peer name veth1
ip link set veth0 netns "$NS"
$ATT ip addr add 192.168.99.1/24 dev veth0
$ATT ip link set veth0 up
$ATT ip link set lo up
ip addr add 192.168.99.2/24 dev veth1
ip link set veth1 up

# The run learns a baseline from this short test; keep the saved profile unchanged
PROFILE=data/traffic_profile.json
[ -f "$PROFILE" ] && cp "$PROFILE" "$PROFILE.before_live_demo"

mkdir -p "$OUT"
# Per-program run time statistics (restored on exit)
BPF_STATS_BEFORE=$(cat /proc/sys/kernel/bpf_stats_enabled 2>/dev/null || echo 0)
echo 1 > /proc/sys/kernel/bpf_stats_enabled 2>/dev/null || echo "bpf_stats not available; XDP cost will be missing"
cleanup() {
    step "Cleaning up"
    [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null || true
    echo "$BPF_STATS_BEFORE" > /proc/sys/kernel/bpf_stats_enabled 2>/dev/null || true
    [ -n "${MAIN_PID:-}" ] && kill "$MAIN_PID" 2>/dev/null || true
    sleep 2
    ip link del veth1 2>/dev/null || true
    ip netns del "$NS" 2>/dev/null || true
    if [ -f "$PROFILE.before_live_demo" ]; then
        mv -f "$PROFILE.before_live_demo" "$PROFILE"
    fi
    echo "Evidence saved in $PROJECT_DIR/$OUT"
}
trap cleanup EXIT

step "Starting the system on veth1 (generic XDP) with dashboard on http://127.0.0.1:${DASH_PORT}"
rm -f data/blacklist.json
"$PY" main.py --interface veth1 --mode generic --dashboard --port "$DASH_PORT" --policy "$POLICY" --models "$MODELS" > "$OUT/system_console.log" 2>&1 &
MAIN_PID=$!
for _ in $(seq 1 30); do
    grep -q "XDP program loaded successfully" "$OUT/system_console.log" 2>/dev/null && break
    if ! kill -0 "$MAIN_PID" 2>/dev/null; then
        echo "The system exited early. Last log lines:"; tail -20 "$OUT/system_console.log"; exit 1
    fi
    sleep 1
done
grep -q "XDP program loaded successfully" "$OUT/system_console.log" || {
    echo "XDP program did not load within 30 s. Last log lines:"; tail -20 "$OUT/system_console.log"; exit 1; }
ip link show veth1 | tee "$OUT/veth1_link.txt" | grep -q xdp && echo "XDP program attached to veth1"

# CPU ticks and RSS of the control plane, once per second: time utime stime rss_kb
( while kill -0 "$MAIN_PID" 2>/dev/null; do
      t=$(date +%s.%N)
      read -r -a st < "/proc/$MAIN_PID/stat" 2>/dev/null || break
      rss=$(awk '/VmRSS/ {print $2}' "/proc/$MAIN_PID/status" 2>/dev/null)
      echo "$t ${st[13]} ${st[14]} ${rss:-0}"
      sleep 1
  done ) > "$OUT/proc_samples.txt" &
SAMPLER_PID=$!

step "Phase 1: ${BASELINE_SECONDS}s of light normal traffic (baseline learning)"
$ATT ping -i 0.2 -c $((BASELINE_SECONDS * 5)) 192.168.99.2 > "$OUT/normal_ping.log" 2>&1 || true

step "Phase 2: ${ATTACK_TYPE} flood from 100 forged bot addresses, ${ATTACK_PPS} pps for ${ATTACK_SECONDS}s (policy ${POLICY})"
ATTACK_START=$(date +%s.%N)
echo y | $ATT "$PY" scripts/attack_simulator.py --type "$ATTACK_TYPE" --distributed --target 192.168.99.2 \
    --interface veth0 --pps "$ATTACK_PPS" --duration "$ATTACK_SECONDS" > "$OUT/attack.log" 2>&1 || true
ATTACK_END=$(date +%s.%N)
sleep 5

step "Collecting evidence"
# XDP program run statistics from the fd held by main.py (prog_type 6 = XDP)
for fd in /proc/"$MAIN_PID"/fdinfo/*; do
    if grep -q "^prog_type:[[:space:]]*6" "$fd" 2>/dev/null; then cat "$fd"; fi
done > "$OUT/xdp_prog_fdinfo.txt" 2>/dev/null || true
CLK_TCK=$(getconf CLK_TCK)
curl -s "http://127.0.0.1:${DASH_PORT}/api/status" > "$OUT/final_status.json" || true
cp logs/alerts.log "$OUT/" 2>/dev/null || true
cp data/blacklist.json "$OUT/" 2>/dev/null || true
"$PY" - "$OUT/final_status.json" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print('Could not read status:', e); sys.exit(0)
st = s.get('statistics', {})
print(f"Total packets seen : {st.get('total_packets', 0):,}")
print(f"Dropped by XDP     : {st.get('dropped_packets', 0):,}")
print(f"Blacklisted sources: {len(s.get('blacklist', []))}")
print(f"Alerts raised      : {len(s.get('recent_alerts', []))}")
print(f"ML enabled         : {s.get('ml_enabled')}")
PY

"$PY" - "$OUT" "$ATTACK_START" "$ATTACK_END" "$CLK_TCK" "$ATTACK_PPS" "$POLICY" "$ATTACK_TYPE" <<'PY'
import json, re, subprocess, sys
from pathlib import Path
out, t_start, t_end, clk, pps, policy, kind = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), \
    int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
m = {'setup': {'mode': 'generic XDP on a veth pair (attacker in its own network namespace)',
               'kernel': subprocess.run(['uname', '-r'], capture_output=True, text=True).stdout.strip(),
               'policy': policy, 'models': __import__('os').environ.get('MODELS', 'synthetic'),
               'attack': kind, 'attack_pps_requested': pps,
               'attack_seconds': round(t_end - t_start, 1)}}
fdinfo = (out / 'xdp_prog_fdinfo.txt').read_text() if (out / 'xdp_prog_fdinfo.txt').exists() else ''
rt = sum(map(int, re.findall(r'run_time_ns:\s*(\d+)', fdinfo)))
rc = sum(map(int, re.findall(r'run_cnt:\s*(\d+)', fdinfo)))
m['xdp'] = {'run_cnt': rc, 'run_time_ns': rt, 'ns_per_packet': rt / rc if rc else None}
rows = [list(map(float, l.split())) for l in (out / 'proc_samples.txt').read_text().split('\n') if l.strip()]
cpu = [100.0 * ((b[1] + b[2]) - (a[1] + a[2])) / clk / (b[0] - a[0]) for a, b in zip(rows, rows[1:]) if b[0] > a[0]]
during = [c for (a, b), c in zip(zip(rows, rows[1:]), cpu) if t_start <= a[0] <= t_end]
m['control_plane'] = {'cpu_pct_mean': sum(cpu) / len(cpu) if cpu else None,
                      'cpu_pct_mean_during_attack': sum(during) / len(during) if during else None,
                      'cpu_pct_max': max(cpu) if cpu else None,
                      'rss_mb_max': max(r[3] for r in rows) / 1024 if rows else None}
try:
    s = json.loads((out / 'final_status.json').read_text())
    tl = [p for p in s.get('timeline', []) if p['t'] >= t_start - 1]
    first_alarm = next((p['t'] for p in tl if p['anomaly']), None)
    first_mit = next((p['t'] for p in tl if p['pps'] > 0 and p['dropped_pps'] >= 0.5 * p['pps']), None)
    att = [p for p in tl if t_start <= p['t'] <= t_end + 1]
    m['detection'] = {'detection_delay_s': None if first_alarm is None else round(first_alarm - t_start, 1),
                      'mitigation_delay_s': None if first_mit is None else round(first_mit - t_start, 1),
                      'attack_seconds_flagged_pct': 100.0 * sum(p['anomaly'] for p in att) / len(att) if att else None,
                      'normal_seconds_flagged_before_attack':
                          sum(p['anomaly'] for p in s.get('timeline', []) if p['t'] < t_start)}
    st = s.get('statistics', {})
    m['packets'] = {k: st.get(k) for k in ('total_packets', 'dropped_packets', 'passed_packets')}
    sent = re.findall(r'Complete: (\d+) packets', (out / 'attack.log').read_text())
    m['packets']['attack_packets_sent'] = sum(map(int, sent)) if sent else None
    if sent and st.get('dropped_packets') is not None:
        m['packets']['dropped_over_attack_sent_pct'] = 100.0 * st['dropped_packets'] / sum(map(int, sent))
    m['note'] = ('timeline covers the last 180 intervals only; dropped_over_attack_sent assumes every '
                 'dropped packet was an attack packet (the only other traffic is ping)')
except Exception as e:  # noqa: BLE001
    m['error'] = f'could not read final status: {e}'
(out / 'live_metrics.json').write_text(json.dumps(m, indent=1))
print(json.dumps(m, indent=1))
PY

if [ "$KEEP_RUNNING" = "1" ]; then
    step "KEEP_RUNNING=1: system stays up for screenshots. Press Ctrl+C to stop."
    wait "$MAIN_PID"
fi
