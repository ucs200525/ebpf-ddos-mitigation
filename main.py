"""
DDoS Mitigation System - command-line entry point

    python main.py --interface Ethernet --dashboard          (Windows: simulation mode)
    sudo python3 main.py --interface eth0 --mode generic --dashboard   (Linux: XDP)

The orchestration logic lives in src/core/system.py.
"""

import argparse
import logging
import sys
import threading

from src.config import NETWORK_INTERFACE, XDP_MODE, MonitoringConfig, MLConfig, PLATFORM
from src.core.system import DDoSMitigationSystem, setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='DDoS Mitigation System - Phase 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on Linux with native XDP
  sudo python main.py --interface eth0
  
  # Run with generic XDP (works on all interfaces)
  sudo python main.py --interface eth0 --mode generic
  
  # Run on Windows
  python main.py --interface "Ethernet"
        """
    )
    
    parser.add_argument(
        '--interface', '-i',
        default=NETWORK_INTERFACE,
        help=f'Network interface to monitor (default: {NETWORK_INTERFACE})'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['native', 'generic', 'offload'],
        default=XDP_MODE,
        help=f'XDP mode (default: {XDP_MODE})'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Enable web dashboard (default port: 5000)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=MonitoringConfig.DASHBOARD_PORT,
        help=f'Dashboard port (default: {MonitoringConfig.DASHBOARD_PORT})'
    )

    parser.add_argument(
        '--host',
        default=MonitoringConfig.DASHBOARD_HOST,
        help=f'Dashboard bind address (default: {MonitoringConfig.DASHBOARD_HOST}; '
             'use 0.0.0.0 to expose on the network)'
    )

    parser.add_argument(
        '--policy',
        choices=['p4', 'top5'],
        default='p4',
        help='Mitigation policy: p4 = source scoring with expiring blocks, rate limits and a '
             'spoofing guard (default); top5 = permanently block the 5 busiest sources per alert'
    )

    parser.add_argument(
        '--enable-sim-api',
        action='store_true',
        help='Allow POST /api/simulate_attack while attached to a live interface '
             '(always enabled in simulation mode). Injected samples can blacklist IPs.'
    )

    # Phase 2: ML arguments
    parser.add_argument(
        '--models',
        choices=['synthetic', 'cic'],
        default='synthetic',
        help='Model set: synthetic (trained on synthetic traffic) or cic (trained on real '
             'CIC-DDoS2019 captures by scripts/cic_experiment.py; also turns on the level-3 flow check)'
    )

    parser.add_argument(
        '--flow-check',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Level 3: verify P4 decisions with the per-flow classifier (default: on with --models cic)'
    )

    parser.add_argument(
        '--ml-model',
        type=str,
        default=None,
        help='Path to trained ML model (enables Phase 2 ML classification)'
    )
    
    parser.add_argument(
        '--train-model',
        action='store_true',
        help='Train a new ML model before starting'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=MLConfig.DATASET_DIR,
        help='Path to CIC-DDoS-2019 dataset (only used by model_trainer --source cic)'
    )
    
    args = parser.parse_args()
    setup_logging()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Check for root/admin privileges
    if PLATFORM == 'linux':
        import os
        if os.geteuid() != 0:
            logger.error("This program must be run with sudo/root privileges on Linux")
            logger.error("Try: sudo python main.py --interface eth0")
            sys.exit(1)
    
    # Handle ML model training if requested
    ml_model_path = args.ml_model
    if args.train_model:
        logger.info("Training ML model...")
        from src.ml.model_trainer import train_model, build_parser
        # Same defaults as `python -m src.ml.model_trainer train`
        train_args = build_parser().parse_args(['train'])

        classifier = train_model(train_args)
        if classifier:
            ml_model_path = train_args.model_path
            logger.info(f"Model trained and saved to {ml_model_path}")
        else:
            logger.warning("Model training failed, continuing without ML")
    
    # Start system
    system = DDoSMitigationSystem(args.interface, args.mode, ml_model_path=ml_model_path,
                                  allow_simulation_api=args.enable_sim_api, policy=args.policy,
                                  models=args.models, flow_check=args.flow_check)

    # Start dashboard in background thread if enabled
    if args.dashboard:
        from src.dashboard.app import run_dashboard
        dashboard_thread = threading.Thread(
            target=run_dashboard,
            args=(system, args.host, args.port),
            daemon=True
        )
        dashboard_thread.start()
        # 127.0.0.1 rather than localhost: on Windows, localhost tries IPv6 first (~2 s delay)
        shown_host = '127.0.0.1' if args.host in ('127.0.0.1', '0.0.0.0') else args.host
        logger.info(f"Dashboard started at http://{shown_host}:{args.port}")
    
    try:
        success = system.start()
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        system.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
