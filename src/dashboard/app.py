"""
Web Dashboard - Flask backend

Serves the single-page dashboard (static/index.html) and the JSON APIs:
  GET  /api/status           current system status (polled by the page)
  POST /api/simulate_attack  inject a traffic sample (simulation/demo only)
  POST /api/demo             play / stop / reset a demo scenario, or switch policy (simulation only)
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import time
from pathlib import Path

from src.config import MonitoringConfig
from src.core.traffic_monitor import validate_blockable_ip

STATIC_DIR = Path(__file__).resolve().parent / 'static'

app = Flask(__name__, static_folder=str(STATIC_DIR))
# The dashboard page is same-origin; only the read-only status API is
# shared cross-origin. The simulation API must never be callable from
# arbitrary web pages.
CORS(app, resources={r"/api/status": {"methods": ["GET"]}})

STAT_FIELDS = (
    'total_packets', 'total_bytes', 'dropped_packets', 'dropped_bytes',
    'passed_packets', 'passed_bytes', 'tcp_packets', 'udp_packets',
    'icmp_packets', 'other_packets',
)
IP_FIELDS = ('packets', 'bytes', 'flow_count', 'syn_count', 'udp_count', 'last_seen')
MAX_SIM_IPS = 5000

# Global reference to DDoS system (set by dashboard runner)
ddos_system = None


@app.route('/')
def index():
    """Dashboard page (static HTML/JS that polls /api/status)"""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/simulate_attack', methods=['POST'])
def api_simulate_attack():
    """
    Inject simulated attack traffic stats into the running system's
    anomaly detector and alert system. Used by run_attack_demo.py.

    Expects JSON body:
    {
      "stats": { total_packets, total_bytes, tcp_packets, udp_packets,
                 icmp_packets, dropped_packets, ... },
      "ip_stats": [ {ip, packets, bytes, flow_count, syn_count, udp_count} ... ],
      "label": "optional label for alert message",
      "interval_seconds": 1.0   (optional: simulated duration of this sample)
    }
    Only available in simulation mode or with main.py --enable-sim-api.
    """
    if not ddos_system or not ddos_system.traffic_monitor:
        return jsonify({'error': 'System not initialized'}), 503
    if not ddos_system.allow_simulation_api:
        return jsonify({'error': 'Simulation API disabled on a live interface '
                                 '(start main.py with --enable-sim-api to allow it)'}), 403

    # No force=True: requiring application/json means browsers must send a
    # CORS preflight, which this endpoint does not grant
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a JSON object body'}), 400

    try:
        stats, ip_stats = _parse_sample(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    label = str(data.get('label', 'simulated'))[:100]

    # Optional simulated clock: each sample covers `interval_seconds`, so
    # computed rates don't depend on HTTP latency between demo posts
    now = None
    interval = data.get('interval_seconds')
    if interval is not None:
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not 0 < interval <= 60:
            return jsonify({'error': "'interval_seconds' must be a number in (0, 60]"}), 400
        last = ddos_system._last_sample_time
        now = (last + interval) if last is not None else time.time()

    alerts_before = len(ddos_system.alert_system.alert_history)
    result = ddos_system.process_sample(stats, ip_stats, label=label, now=now)

    return jsonify({
        'is_anomaly': result.is_anomaly,
        'score': result.score,
        'reasons': result.reasons,
        'detection_source': result.detection_source,
        'attack_type': result.attack_type,
        'ml_confidence': result.ml_confidence,
        'alert_fired': len(ddos_system.alert_system.alert_history) > alerts_before,
        'baseline_pps': ddos_system.anomaly_detector.baseline.mean_pps,
        'blacklist_size': len(ddos_system.traffic_monitor.get_blacklist()),
    })


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or value != value:
        raise ValueError(f"'{name}' must be a non-negative number")
    return int(value)


def _parse_sample(data: dict):
    """Validate a simulated sample: cumulative stats + per-interval IP counts"""
    raw_stats = data.get('stats')
    if not isinstance(raw_stats, dict) or not raw_stats:
        raise ValueError("'stats' must be a non-empty object")
    stats = {k: _non_negative_int(raw_stats.get(k, 0), k) for k in STAT_FIELDS}

    raw_ips = data.get('ip_stats', [])
    if not isinstance(raw_ips, list) or len(raw_ips) > MAX_SIM_IPS:
        raise ValueError(f"'ip_stats' must be a list of at most {MAX_SIM_IPS} entries")
    ip_stats = []
    for entry in raw_ips:
        if not isinstance(entry, dict) or not isinstance(entry.get('ip'), str):
            raise ValueError("each ip_stats entry needs an 'ip' string")
        ip = validate_blockable_ip(entry['ip'])
        if ip is None:
            raise ValueError(f"invalid or protected IP address: {entry['ip'][:50]!r}")
        parsed = {'ip': ip}
        for field in IP_FIELDS:
            parsed[field] = _non_negative_int(entry.get(field, 0), field)
        ip_stats.append(parsed)
    return stats, ip_stats


def _ensure_demo():
    """Attach the demo engine once the system runs in simulation mode
    (created lazily: the dashboard starts before the monitor exists)"""
    system = ddos_system
    monitor = system.traffic_monitor if system else None
    if system and system.demo is None and monitor and monitor.simulation_mode             and system.allow_simulation_api:
        from src.dashboard.demo import DemoRunner
        system.demo = DemoRunner(system)


@app.route('/api/demo', methods=['POST'])
def api_demo():
    """
    Control the built-in demo (simulation mode only).
    JSON body: {"action": "play", "scenario": "udp_flood"} | {"action": "stop"} | {"action": "reset"}
    | {"action": "policy", "policy": "p4" or "top5"} (resets the demo, then switches)
    """
    _ensure_demo()
    if not ddos_system or not ddos_system.demo:
        return jsonify({'error': 'Demo is only available in simulation mode'}), 403
    # JSON only: browsers must preflight, which this endpoint does not grant cross-site
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a JSON object body'}), 400
    action = data.get('action')
    demo = ddos_system.demo
    if action == 'play':
        error = demo.play(str(data.get('scenario', ''))[:40])
        if error:
            return jsonify({'error': error}), 400
    elif action == 'stop':
        demo.stop()
    elif action == 'reset':
        demo.reset()
    elif action == 'policy':
        # Switch the mitigation policy so the audience can compare them on the same scenario
        policy = str(data.get('policy', ''))[:10]
        if policy not in ('p4', 'top5'):
            return jsonify({'error': "'policy' must be p4 or top5"}), 400
        demo.reset()
        ddos_system.set_policy(policy)
    else:
        return jsonify({'error': "'action' must be play, stop, reset or policy"}), 400
    return jsonify(demo.status())


@app.route('/api/status')
def api_status():
    """Get current system status"""
    if ddos_system:
        _ensure_demo()
        return jsonify(ddos_system.get_status())
    return jsonify({'error': 'System not initialized'}), 503


def run_dashboard(system, host=MonitoringConfig.DASHBOARD_HOST, port=MonitoringConfig.DASHBOARD_PORT):
    """
    Run the dashboard server
    
    Args:
        system: DDoSMitigationSystem instance
        host: Host to bind to
        port: Port to listen on
    """
    global ddos_system
    ddos_system = system

    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    print("Dashboard should be run via main.py with --dashboard flag")
