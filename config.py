import yaml
import argparse
from pathlib import Path
import torch
import collections


def config(args=None):
    """
    Reads the command switches and creates a config
    Command line switches override config files
    :return:
    """

    """ config """
    parser = argparse.ArgumentParser(description='configuration switches')
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('--transfer_load', type=str, default=None)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--processes', type=int)
    parser.add_argument('--seed', type=int, default=None)

    """ visualization params """
    parser.add_argument('--display', type=int)

    """ model parameters """
    parser.add_argument('--optlevel', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_in_channels', type=int)
    parser.add_argument('--model_keypoints', type=int)

    """ hyper-parameters """
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--lr', type=float)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_train_len', type=int)
    parser.add_argument('--dataset_test_len', type=int)
    parser.add_argument('--dataset_randomize', type=int)

    args = parser.parse_args(args)

    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def set_if_not_set(args, dict):
        for key, value in dict.items():
            if key in vars(args) and vars(args)[key] is None:
                vars(args)[key] = dict[key]
            elif key not in vars(args):
                vars(args)[key] = dict[key]
        return args

    if args.config is not None:
        with Path(args.config).open() as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            conf = flatten(conf)
            args = set_if_not_set(args, conf)

    defaults = {
        'optimizer': 'Adam',
        'lr': 1e-4,
        'checkpoint_freq': 1000,
        'opt_level': 'O0',
        'display_kp_rows': 4,
        'display_freq': 5000,
    }

    args = set_if_not_set(args, defaults)

    if args.device is None:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    def counter():
        run_id_pid = Path('./.run_id')
        count = 1
        if run_id_pid.exists():
            with run_id_pid.open('r+') as f:
                last_id = int(f.readline())
                last_id += 1
                count = last_id
                f.seek(0)
                f.write(str(last_id))
        else:
            with run_id_pid.open('w+') as f:
                f.write(str(count))
        return count

    if args.run_id == -1:
        args.run_id = counter()

    return args
