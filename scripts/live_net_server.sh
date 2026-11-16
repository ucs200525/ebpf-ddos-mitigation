#!/usr/bin/env bash
# Protection server on the REAL network (Linux / WSL2 Kali, root).
#
# Attaches the system to the interface that faces the Wi-Fi hotspot, in generic XDP mode,
# with the dashboard open to the network. Another machine on the same hotspot floods this
# host (project/attacker/), the system blacklists that source, and XDP drops its packets
# at the driver hook. The gateway and this host are protected so the network stays up.
#
#     sudo bash scripts/live_net_server.sh          (Windows: double-click start_server.bat)
# Options:  PORT=5000  POLICY=p4|top5  MODELS=synthetic|cic  IFACE=eth5 (auto-detected)
set -euo pipefail

PORT=${PORT:-5000}
POLICY=${POLICY:-p4}
MODELS=${MODELS:-synthetic}

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
[ "$(id -u)" -eq 0 ] || { echo "Run as root:  sudo bash scripts/live_net_server.sh"; exit 1; }

step() { echo; echo "==> $*"; }

# Interface that holds the default route = the one on the hotspot
IFACE=${IFACE:-$(ip -4 route show default | awk '{print $5; exit}')}
SERVER_IP=$(ip -4 addr show "$IFACE" | awk '/inet /{print $2; exit}' | cut -d/ -f1)
GATEWAY=$(ip -4 route show default | awk '{print $3; exit}')
[ -n "$IFACE" ] && [ -n "$SERVER_IP" ] || { echo "Could not find the hotspot interface. Connect to the hotspot first."; exit 1; }

step "Checking dependencies"
missing=""
python3 -c "import bcc" 2>/dev/null || missing="$missing python3-bpfcc"
command -v ip >/dev/null || missing="$missing iproute2"
command -v curl >/dev/null || missing="$missing curl"
[ -n "$missing" ] && { echo "Installing:$missing"; apt-get update -qq && apt-get install -y -qq $missing; }
if [ ! -x .venv-live/bin/python ]; then
    step "Creating Python environment (.venv-live, first run only)"
    python3 -m venv --system-site-packages .venv-live
fi
.venv-live/bin/pip install -q -r requirements.txt
PY="$PROJECT_DIR/.venv-live/bin/python"

PROFILE=data/traffic_profile.json
[ -f "$PROFILE" ] && cp "$PROFILE" "$PROFILE.before_livenet"
mkdir -p logs
LOG="logs/live_net_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
    step "Stopping"
    [ -n "${MAIN_PID:-}" ] && kill "$MAIN_PID" 2>/dev/null || true
    sleep 2
    # Make sure no XDP program is left on the interface
    ip link set dev "$IFACE" xdpgeneric off 2>/dev/null || true
    ip link set dev "$IFACE" xdp off 2>/dev/null || true
    pkill -f "tail -n 0 -f $LOG" 2>/dev/null || true
    [ -f "$PROFILE.before_livenet" ] && mv -f "$PROFILE.before_livenet" "$PROFILE"
    echo "Log: $PROJECT_DIR/$LOG"
}
trap cleanup EXIT
trap "exit 130" INT TERM HUP

step "Protecting $IFACE ($SERVER_IP) with generic XDP; gateway $GATEWAY and this host are never blocked"
rm -f data/blacklist.json
"$PY" main.py --interface "$IFACE" --mode generic --dashboard --host 0.0.0.0 --port "$PORT" \
    --policy "$POLICY" --models "$MODELS" --protect "$GATEWAY,$SERVER_IP" > "$LOG" 2>&1 &
MAIN_PID=$!
for _ in $(seq 1 60); do
    grep -q "XDP program loaded successfully" "$LOG" 2>/dev/null && break
    kill -0 "$MAIN_PID" 2>/dev/null || { echo "The system exited early:"; tail -25 "$LOG"; exit 1; }
    sleep 1
done
grep -q "XDP program loaded successfully" "$LOG" || { echo "XDP did not load:"; tail -25 "$LOG"; exit 1; }
ip link show "$IFACE" | grep -q xdp && echo "XDP program attached to $IFACE (kernel $(uname -r))"

echo
echo "================================================================"
echo "  SERVER READY - live XDP on $IFACE, kernel $(uname -r)"
echo
echo "  Monitor (phone / projector):  http://$SERVER_IP:$PORT/"
echo "  On this laptop:               http://127.0.0.1:$PORT/"
echo
echo "  ON THE ATTACKER LAPTOP, aim the attack tool at:  $SERVER_IP"
echo "    Windows:  attacker\\attack.bat $SERVER_IP"
echo "    Linux:    sudo bash attacker/attack.sh $SERVER_IP"
echo
echo "  Protected (never blocked): gateway $GATEWAY, this host $SERVER_IP"
echo "  Ctrl+C stops the server and removes XDP from $IFACE."
echo "================================================================"
echo
tail -n 0 -f "$LOG" --pid="$MAIN_PID" | grep --line-buffered -E "Status:|ATTACK|Added .* to blacklist|Alert|WARNING|ERROR" &
wait "$MAIN_PID" || true
