from io_basenet import load_yaml, save_yaml
from seed_basenet import seed_everything
from logger_basenet import setup_logging

def load_config(config_path):
    cfg = load_yaml(config_path)
    out = cfg.setdefault('output', {})
    out.setdefault('base_dir', 'basenet_1km')
    out.setdefault('logs_dir', 'good_archi/logs')
    out.setdefault('results_dir', 'good_archi/results')
    out.setdefault('checkpoints_dir', 'good_archi/checkpoints')
    return cfg

