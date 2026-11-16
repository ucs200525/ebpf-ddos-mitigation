#!/usr/bin/env bash
# DDoS attack tool for the capstone demo (attacker laptop, Linux).
# Usage:  sudo bash attack.sh [server-ip] [type]     e.g.  sudo bash attack.sh 10.23.12.77 udp
# Root is needed for the fast ICMP flood and the spoofed SYN flood.
cd "$(dirname "$0")"
PYES=python3; command -v python3 >/dev/null || PYES=python
exec "$PYES" ddos_attack.py "$@"
