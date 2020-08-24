import logging
import argparse
import yaml

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

TRAIN_ARGS = [
    'batch_size',
    'lr',
    'n_epochs',
    'clip',
    'min_freq',
]


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_file_logger(log_file_path):
    formatter = logging.Formatter(LOG_FORMAT)
    fileh = logging.FileHandler(log_file_path, 'a')
    fileh.setFormatter(formatter)

    log = logging.getLogger()
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)
    log.addHandler(fileh)
