"""
Project layout checks: paths come from src/config.py and resolve inside the
project folder, independent of the current working directory.
"""

from pathlib import Path

from src import config
from src.config import AlertConfig, MLConfig, MonitoringConfig, ProfilingConfig


def test_paths_are_absolute_and_inside_project():
    root = config.PROJECT_ROOT
    assert (root / 'main.py').is_file()
    for path in (config.EBPF_PROGRAM_PATH, config.BLACKLIST_FILE, MLConfig.DEFAULT_MODEL_PATH,
                 MLConfig.DATASET_DIR, MonitoringConfig.LOG_FILE, AlertConfig.ALERT_FILE,
                 ProfilingConfig.PROFILE_FILE):
        p = Path(path)
        assert p.is_absolute(), path
        assert root in p.parents, path


def test_required_assets_present():
    assert Path(config.EBPF_PROGRAM_PATH).is_file()
    assert (config.SRC_DIR / 'ebpf' / 'xdp_filter_libbpf.c').is_file()
    assert (config.SRC_DIR / 'dashboard' / 'static' / 'index.html').is_file()
    assert Path(MLConfig.DEFAULT_MODEL_PATH).is_file()


def test_provenance_records_code_environment_and_inputs():
    from src.provenance import file_sha256, provenance
    p = provenance('test', {'seed': 1}, {'model': MLConfig.DEFAULT_MODEL_PATH})
    assert p['parameters'] == {'seed': 1} and p['scikit_learn'] and p['python']
    assert p['inputs']['model']['sha256'] == file_sha256(MLConfig.DEFAULT_MODEL_PATH)
    assert p['inputs']['model']['path'].replace('\\', '/') == 'data/models/ddos_classifier.joblib'


def test_works_from_another_directory(tmp_path, monkeypatch):
    """Components must not depend on the current directory"""
    monkeypatch.chdir(tmp_path)
    from src.core.anomaly_detector import MLEnhancedAnomalyDetector
    detector = MLEnhancedAnomalyDetector(model_path=MLConfig.DEFAULT_MODEL_PATH)
    assert detector.ml_enabled
    assert not (tmp_path / 'data').exists()
