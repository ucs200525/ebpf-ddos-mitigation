#!/usr/bin/env python3
"""
Compile-check the XDP program with BCC (Linux only).

As a normal user BCC compiles the C source and then fails when it asks the
kernel to create maps (needs root): reaching that point means the program
compiled. Run with sudo to also create the maps and load the program through
the kernel verifier (it is not attached to any interface).

    python3 scripts/check_xdp_compile.py           # compile only
    sudo python3 scripts/check_xdp_compile.py      # compile + verifier
"""

import os
import sys
from pathlib import Path

from bcc import BPF

SOURCE = Path(__file__).resolve().parent.parent / 'src' / 'ebpf' / 'xdp_filter.c'

try:
    bpf = BPF(src_file=str(SOURCE).encode(), cflags=['-w'])
    bpf.load_func('xdp_ddos_filter', BPF.XDP)
    print('OK: compiled, maps created and program accepted by the kernel verifier')
except Exception as exc:  # noqa: BLE001 - report whatever BCC raised
    text = str(exc)
    if os.geteuid() != 0 and ('Operation not permitted' in text or 'Failed to create' in text
                              or 'Permission denied' in text):
        print('OK: compiled (map creation needs root; run with sudo to check the verifier too)')
    else:
        print(f'FAILED: {type(exc).__name__}: {text}')
        sys.exit(1)
