#!/usr/bin/env bash
# Protection server with live XDP and a traffic lab (Linux / WSL2 Kali, root).
#
#   veth0 (192.168.99.1, namespace ddos_attacker)  ->  veth1 (192.168.99.2, XDP attached)
#
# Starts the system on veth1 in generic XDP mode with the dashboard on all addresses, so a
# phone and a second laptop on the same network can open:
#     http://<this machine>:5000/          live monitor
#     http://<this machine>:5000/console   attack console (sends real packets into veth0)
# Normal traffic from 80 regular clients starts automatically so the baseline is learned.
#
#     sudo bash scripts/lab_server.sh            (Windows: double-click start_server.bat)
# Options:  PORT=5000  POLICY=p4|top5  MODELS=synthetic|cic
set -euo pipefail

PORT=${PORT:-5000}
POLICY=${POLICY:-p4}
MODELS=${MODELS:-synthetic}
NS=ddos_attacker

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
[ "$(id -u)" -eq 0 ] || { echo "Run as root:  sudo bash scripts/lab_server.sh"; exit 1; }

step() { echo; echo "==> $*"; }

step "Checking dependencies"
missing=""
python3 -c "import bcc" 2>/dev/null || missing="$missing python3-bpfcc"
command -v ip >/dev/null || missing="$missing iproute2"
command -v curl >/dev/null || missing="$missing curl"
if [ -n "$missing" ]; then
    echo "Installing:$missing"
    apt-get update -qq && apt-get install -y -qq $missing
fi
if [ ! -x .venv-live/bin/python ]; then
    step "Creating Python environment (.venv-live, first run only)"
    python3 -m venv --system-site-packages .venv-live
fi
.venv-live/bin/pip install -q -r requirements.txt
PY="$PROJECT_DIR/.venv-live/bin/python"

step "Creating the lab link: veth0 (attacker side, namespace $NS) <-> veth1 (protected, XDP)"
ip link del veth1 2>/dev/null || true
ip netns del "$NS" 2>/dev/null || true
ip netns add "$NS"
ip link add veth0 type veth peer name veth1
ip link set veth0 netns "$NS"
ip netns exec "$NS" ip addr add 192.168.99.1/24 dev veth0
ip netns exec "$NS" ip link set veth0 up
ip netns exec "$NS" ip link set lo up
ip addr add 192.168.99.2/24 dev veth1
ip link set veth1 up
# Replies to the lab's client and bot addresses go back into the lab, never onto the real network
for net in 198.18.0.0/15 45.32.0.0/14; do ip route replace "$net" dev veth1; done

PROFILE=data/traffic_profile.json
[ -f "$PROFILE" ] && cp "$PROFILE" "$PROFILE.before_lab"
mkdir -p logs
LOG="logs/lab_server_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
    step "Stopping"
    [ -n "${MAIN_PID:-}" ] && kill "$MAIN_PID" 2>/dev/null || true
    sleep 2
    pkill -f "scripts/lab_traffic.py" 2>/dev/null || true
    pkill -f "tail -n 0 -f $LOG" 2>/dev/null || true
    for net in 198.18.0.0/15 45.32.0.0/14; do ip route del "$net" dev veth1 2>/dev/null || true; done
    ip link del veth1 2>/dev/null || true
    ip netns del "$NS" 2>/dev/null || true
    [ -f "$PROFILE.before_lab" ] && mv -f "$PROFILE.before_lab" "$PROFILE"
    echo "Log: $PROJECT_DIR/$LOG"
}
trap cleanup EXIT
trap "exit 130" INT TERM HUP

step "Starting the protection system on veth1 (generic XDP), dashboard on port $PORT"
rm -f data/blacklist.json
"$PY" main.py --interface veth1 --mode generic --dashboard --host 0.0.0.0 --port "$PORT" \
    --policy "$POLICY" --models "$MODELS" --lab > "$LOG" 2>&1 &
MAIN_PID=$!
for _ in $(seq 1 60); do
    grep -q "XDP program loaded successfully" "$LOG" 2>/dev/null && break
    kill -0 "$MAIN_PID" 2>/dev/null || { echo "The system exited early:"; tail -20 "$LOG"; exit 1; }
    sleep 1
done
grep -q "XDP program loaded successfully" "$LOG" || { echo "XDP did not load:"; tail -20 "$LOG"; exit 1; }
ip link show veth1 | grep -q xdp && echo "XDP program attached to veth1 (kernel $(uname -r))"

step "Starting normal traffic (80 regular clients) so the baseline is learned"
for _ in $(seq 1 30); do
    curl -sf "http://127.0.0.1:$PORT/api/status" >/dev/null && break
    sleep 1
done
curl -sf -X POST -H 'Content-Type: application/json' -d '{"action":"play","scenario":"normal"}' \
    "http://127.0.0.1:$PORT/api/demo" >/dev/null || echo "Could not start normal traffic; use the console"

LAN_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<NF;i++) if ($i=="src") print $(i+1)}')
echo
echo "================================================================"
echo "  Server running: live XDP on veth1, kernel $(uname -r)"
echo "  Monitor (phone / projector):  http://${LAN_IP:-<this-ip>}:$PORT/"
echo "  Attack console (laptop 2):    http://${LAN_IP:-<this-ip>}:$PORT/console"
echo "  On this laptop:               http://127.0.0.1:$PORT/"
echo "  Ctrl+C stops the server and removes the lab link."
echo "================================================================"
echo
# Show the system's own status lines while it runs
# (in the background, so Ctrl+C / closing the window reaches the cleanup trap at once)
tail -n 0 -f "$LOG" --pid="$MAIN_PID" | grep --line-buffered -E "Status:|ATTACK|Alert|WARNING|ERROR" &
wait "$MAIN_PID" || true
