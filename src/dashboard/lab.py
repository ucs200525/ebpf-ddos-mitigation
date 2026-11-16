"""
Traffic lab for live XDP (Linux, root): the dashboard's traffic profiles as real packets.

scripts/lab_server.sh creates a veth pair: veth0 in the network namespace "ddos_attacker"
and veth1 in the server namespace, where the XDP program is attached. LabRunner keeps two
helper processes (scripts/lab_traffic.py):

  sender    inside the attacker namespace: sends real frames from the client, crowd and
            bot address ranges into veth0, following the profile the console picks
  receiver  on veth1: counts the frames that get past XDP

Detection, blocking and dropping are the kernel's and the system's own; the only thing the
lab knows that the system does not is which addresses are bots, used for the
"attack dropped / legitimate lost" line (sent minus received, per side).
"""

import ipaddress
import subprocess
import sys
import threading
from pathlib import Path

from src.dashboard.demo import DemoRunner

# Address plan (scripts/lab_server.sh routes these ranges back into veth1):
#   regular clients and flash crowd: 198.18.0.0/15 (RFC 2544 benchmarking range)
#   bots: 45.32.0.0/14
CLIENTS = [f'198.18.{i // 250}.{i % 250 + 1}' for i in range(80)]
CROWD = [f'198.19.{i // 250}.{i % 250 + 1}' for i in range(1500)]
BOTS = [f'45.{33 + i // 250}.{(i * 7) % 250 + 1}.{i % 250 + 1}' for i in range(400)]
BOT_NET = ipaddress.ip_network('45.32.0.0/14')

LAB_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'lab_traffic.py'


def is_attack_ip(ip: str) -> bool:
    return ipaddress.ip_address(ip) in BOT_NET


class LabRunner(DemoRunner):
    """Same controls and status as DemoRunner, but the traffic is real packets"""

    def __init__(self, system, netns: str = 'ddos_attacker', send_iface: str = 'veth0',
                 recv_iface: str = 'veth1', target: str = '192.168.99.2'):
        super().__init__(system)
        self.clients, self.crowd, self.bots = CLIENTS, CROWD, BOTS
        self._sent = [0, 0]     # legit, attack since the last reset
        self._recv = [0, 0]
        self._cmd_lock = threading.Lock()
        py = sys.executable
        self.sender = subprocess.Popen(
            ['ip', 'netns', 'exec', netns, py, str(LAB_SCRIPT), 'send', '--iface', send_iface, '--target', target],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.receiver = subprocess.Popen(
            [py, str(LAB_SCRIPT), 'recv', '--iface', recv_iface],
            stdout=subprocess.PIPE, text=True, bufsize=1)
        for proc in (self.sender, self.receiver):
            threading.Thread(target=self._read, args=(proc,), daemon=True).start()
        self._sent_profile = None

    # ---- helper processes ---------------------------------------------
    def _read(self, proc):
        for line in proc.stdout:
            parts = line.split()
            if len(parts) == 3 and parts[0] in ('sent', 'recv'):
                target = self._sent if parts[0] == 'sent' else self._recv
                target[0] += int(parts[1])
                target[1] += int(parts[2])
                self._update_outcome()

    def _update_outcome(self):
        (sl, sa), (rl, ra) = self._sent, self._recv
        self.outcome = dict(legit=sl, legit_dropped=max(0, sl - rl), attack=sa, attack_dropped=max(0, sa - ra))

    def _command(self, profile: str):
        with self._cmd_lock:
            if profile == self._sent_profile:
                return
            self._sent_profile = profile
            try:
                self.sender.stdin.write(profile + '\n')
                self.sender.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    # ---- DemoRunner hooks ---------------------------------------------
    def _next(self):
        super()._next()
        self._command(self._current or 'idle')

    def _feed(self, scenario: str, t: int):
        # The sender produces the packets; the monitoring loop reads the kernel counters
        self._command(scenario)

    def stop(self):
        # The regular clients keep sending: a protected server never sees zero traffic,
        # and silent seconds would drag the learned baseline down
        super().stop()
        self.play('normal')

    def reset(self):
        super().reset()
        self._sent, self._recv = [0, 0], [0, 0]
        self._update_outcome()

    def shutdown(self):
        super().shutdown()
        for proc in (self.sender, self.receiver):
            try:
                proc.terminate()
            except OSError:
                pass
