#!/usr/bin/env python3
"""
One-step launcher for Windows and Linux (use run.bat or run.sh, or `python run.py`).

The first run creates a virtual environment (.venv-windows or .venv-linux, so one folder
works from both Windows and WSL) and installs requirements.txt;
later runs reinstall only when requirements.txt changes. Then it runs one action:

  demo        Dashboard + built-in demo, opens the browser   (Windows: simulation mode;
              Linux: needs sudo and attaches XDP, see `dashboard`)
  dashboard   Linux only: run the system on a real interface with XDP, e.g.
                ./run.sh dashboard eth0          (asks for the sudo password)
  live        Linux only: full scripted live XDP test on a veth pair (sudo)
  check-xdp   Linux only: compile the XDP program and load it through the kernel verifier (sudo)
  attack      Send the scripted demo attack to a dashboard that is already running
  test        Run the test suite
  verify      Check every number quoted in the paper/report/slides against the result files
  reproduce   Rerun every experiment, figure, check and test (about 25 min on 16 cores)
  figures     Regenerate figures and result tables from the saved result files
  cic DIR     Real-attack pipeline on downloaded CIC-DDoS2019 zips (DIR holds CSVs/ and PCAPs/):
              inventory of the labels, extraction of the captures, training + tests of all
              three levels (about 30 min; needs ~2 GB free disk, nothing is unzipped)
  live-cic DIR  Linux only (root): real CIC-DDoS2019 test-day packets replayed through the
              system in generic XDP mode, four configurations side by side; kernel-measured
              results in data/results/cic_xdp_results.json (about 50 min; needs `run.sh cic`
              first, plus tcpreplay, tcpdump and bpftool)
  check-flows Linux only (root): send known packets through XDP and compare every kernel counter
  setup       Only create/update the environment

Examples:
  run.bat                     (Windows: same as `run.bat demo`)
  ./run.sh test
  ./run.sh dashboard eth0
  run.bat cic H:/CICDDoS2019
  run.bat demo --models cic   (dashboard with the models trained on real captures)
"""

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
# One environment per OS: the same folder can be used from Windows and from WSL
VENV = ROOT / ('.venv-windows' if IS_WINDOWS else '.venv-linux')
REQUIREMENTS = ROOT / 'requirements.txt'
MARKER = VENV / '.requirements.sha256'
DASHBOARD_URL = 'http://127.0.0.1:5000'


def say(msg: str) -> None:
    print(f'[run] {msg}', flush=True)


def venv_python() -> Path:
    return VENV / ('Scripts/python.exe' if IS_WINDOWS else 'bin/python')


def setup() -> Path:
    """Create .venv and install requirements if they changed since the last install"""
    if sys.version_info < (3, 10):
        sys.exit(f'Python 3.10 or newer is needed (this is {platform.python_version()}).')
    py = venv_python()
    if not py.exists():
        say(f'Creating virtual environment in {VENV.name} (first run only)')
        # On Linux the BCC module (python3-bpfcc) is a system package, so the
        # environment must also see system site-packages
        try:
            venv.EnvBuilder(with_pip=True, system_site_packages=IS_LINUX).create(VENV)
        except Exception as e:  # noqa: BLE001
            hint = ' Install it with: sudo apt install python3-venv' if IS_LINUX else ''
            sys.exit(f'Could not create the virtual environment: {e}.{hint}')
    wanted = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not MARKER.exists() or MARKER.read_text().strip() != wanted:
        say('Installing requirements (this takes a few minutes the first time)')
        subprocess.run([str(py), '-m', 'pip', 'install', '--upgrade', '-q', 'pip'], check=False)
        rc = subprocess.run([str(py), '-m', 'pip', 'install', '-q', '-r', str(REQUIREMENTS)]).returncode
        if rc != 0:
            sys.exit('Installing requirements failed; see the messages above.')
        MARKER.write_text(wanted)
    return py


def run(cmd, **kw) -> int:
    say(' '.join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw).returncode


def need_linux(action: str) -> None:
    if not IS_LINUX:
        sys.exit(f'`{action}` needs Linux with eBPF (on Windows use `run.bat demo`).')


def sudo_prefix() -> list:
    if os.geteuid() == 0:
        return []
    if not shutil.which('sudo'):
        sys.exit('This action needs root and sudo is not installed.')
    say('Root is needed for XDP: sudo will ask for your password')
    return ['sudo', '-E']


def wait_and_open_browser(proc, url: str, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            urllib.request.urlopen(url + '/api/status', timeout=2)
            say(f'Dashboard ready: {url}  (press "Guided tour"; Ctrl+C here stops it)')
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(1)
    if proc.poll() is None:
        say(f'Dashboard not answering yet; open {url} yourself')


def run_dashboard(py: Path, args: list, root: bool = False) -> int:
    cmd = (sudo_prefix() if root else []) + [str(py), 'main.py', *args]
    say(' '.join(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        wait_and_open_browser(proc, DASHBOARD_URL)
        return proc.wait()
    except KeyboardInterrupt:
        say('Stopping...')
        proc.terminate()
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 1


def main() -> int:
    argv = sys.argv[1:]
    action = argv[0].lower() if argv else 'demo'
    rest = argv[1:]
    if action in ('-h', '--help', 'help'):
        print(__doc__)
        return 0

    actions = ('setup', 'demo', 'dashboard', 'live', 'check-xdp', 'attack', 'test', 'verify', 'reproduce',
               'figures', 'cic', 'live-cic', 'check-flows')
    if action not in actions:
        print(__doc__)
        return 2

    if action == 'live':
        # live_xdp_demo.sh builds and uses its own environment (.venv-live) as root
        need_linux(action)
        return run(sudo_prefix() + ['bash', 'scripts/live_xdp_demo.sh', *rest])
    if action == 'live-cic':
        need_linux(action)
        if not rest:
            sys.exit('Give the folder with the downloaded CIC-DDoS2019 zips, e.g. ./run.sh live-cic /mnt/h/CICDDoS2019')
        root, replay = sudo_prefix(), '/root/cic_replay'
        out = '/root/cic_xdp_runs/' + time.strftime('%Y%m%d_%H%M%S')
        live_py = str(ROOT / '.venv-live' / 'bin' / 'python')
        for cmd in ([live_py, 'scripts/cic_replay_pcap.py', '--cic-dir', rest[0], '--out', replay],
                    ['bash', 'scripts/cic_xdp_replay.sh', replay, out],
                    [live_py, 'scripts/cic_xdp_analyze.py', out, '--replay-dir', replay]):
            rc = run(root + cmd)
            if rc != 0:
                return rc
        return 0
    if action == 'check-flows':
        need_linux(action)
        return run(sudo_prefix() + [str(ROOT / '.venv-live' / 'bin' / 'python'), 'scripts/check_xdp_flows.py'])

    py = setup()
    if action == 'setup':
        say('Environment ready.')
        return 0
    if action == 'demo':
        if IS_LINUX:
            say('On Linux the system attaches XDP and needs root. Use `./run.sh dashboard <interface>` '
                'for a real interface, or `./run.sh live` for the scripted veth test.')
            return 2
        return run_dashboard(py, ['--interface', 'Ethernet', '--dashboard', *rest])
    if action == 'dashboard':
        need_linux(action)
        if not rest:
            sys.exit('Give the interface to protect, e.g. ./run.sh dashboard eth0 '
                     '(list them with: ip -br link)')
        iface, extra = rest[0], rest[1:]
        return run_dashboard(py, ['--interface', iface, '--mode', 'generic', '--dashboard', *extra], root=True)
    if action == 'check-xdp':
        need_linux(action)
        return run(sudo_prefix() + [py, 'scripts/check_xdp_compile.py'])
    if action == 'attack':
        return run([py, 'scripts/run_attack_demo.py', *rest])
    if action == 'test':
        return run([py, '-m', 'pytest', '-q', *rest])
    if action == 'verify':
        return run([py, 'scripts/verify_results.py', *rest])
    if action == 'reproduce':
        return run([py, 'scripts/reproduce_all.py', *rest])
    if action == 'figures':
        return run([py, 'scripts/make_figures.py', *rest])
    if action == 'cic':
        if not rest:
            sys.exit('Give the folder with the downloaded CIC-DDoS2019 zips, e.g. run.bat cic H:/CICDDoS2019')
        cic_dir, extra = rest[0], rest[1:]
        for cmd in (['scripts/cic_inspect.py', '--cic-dir', cic_dir],
                    ['scripts/cic_extract.py', '--cic-dir', cic_dir],
                    ['scripts/cic_experiment.py', *extra]):
            rc = run([py, *cmd])
            if rc != 0:
                return rc
        return 0
    return 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
