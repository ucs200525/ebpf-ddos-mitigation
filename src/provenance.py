"""
Provenance block stored in every result file: which code, environment and
parameters produced it, so each reported number can be traced to a run.
"""

import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _git(*args) -> Optional[str]:
    try:
        out = subprocess.run(['git', *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_sha256() -> str:
    """One hash over every source file that can change a result (src/, scripts/, eBPF C)"""
    h = hashlib.sha256()
    files = sorted(p for d in ('src', 'scripts') for p in (PROJECT_ROOT / d).rglob('*')
                   if p.suffix in ('.py', '.c', '.h', '.sh') and '__pycache__' not in p.parts)
    for p in files:
        h.update(str(p.relative_to(PROJECT_ROOT)).replace('\\', '/').encode())
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def provenance(script: str, parameters: Dict, inputs: Optional[Dict[str, str]] = None) -> Dict:
    """
    Args:
        script: the script or command that produced the result
        parameters: every parameter that changes the result (seeds, run counts, ...)
        inputs: name -> path of model/data files the result depends on (hashed)
    """
    import numpy
    import sklearn
    # git runs with cwd = project/, so '.' is the project folder
    status = _git('status', '--porcelain', '--', '.')
    return {
        'script': script,
        'parameters': parameters,
        'created': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'git_commit': _git('rev-parse', 'HEAD'),
        # Uncommitted changes under project/ when the run started (None = git unavailable)
        'git_dirty': None if status is None else bool(status),
        # Identifies the exact code even when it is not committed
        'code_sha256': code_sha256(),
        'python': sys.version.split()[0],
        'numpy': numpy.__version__,
        'scikit_learn': sklearn.__version__,
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'inputs': {name: {'path': str(Path(path).resolve().relative_to(PROJECT_ROOT))
                          if Path(path).resolve().is_relative_to(PROJECT_ROOT) else str(path),
                          'sha256': file_sha256(path)}
                   for name, path in (inputs or {}).items()},
    }
