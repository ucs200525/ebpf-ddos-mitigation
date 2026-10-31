#!/usr/bin/env bash
# One-step launcher for Linux:
#   ./run.sh test                 run the tests
#   ./run.sh verify               check every number in the paper/report/slides
#   ./run.sh dashboard eth0       run the system with XDP on eth0 + dashboard (sudo)
#   ./run.sh live                 scripted live XDP test on a veth pair (sudo)
#   ./run.sh check-xdp            compile + kernel-verifier check of the XDP program (sudo)
#   ./run.sh reproduce            rerun every experiment (about 25 min)
#   ./run.sh help                 all actions
# The first run creates .venv-linux and installs requirements.txt (needs Python 3.10+ and python3-venv).
# For XDP install BCC first:  sudo apt install python3-bpfcc bpfcc-tools linux-headers-$(uname -r)
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
    echo "Python 3.10+ is needed: sudo apt install python3 python3-venv" >&2
    exit 1
fi
exec "$PY" run.py "$@"
