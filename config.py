import yaml
import argparse
from pathlib import Path
import torch
import collections
import re


class NullScheduler:
    """ Empty scheduler for use as a placeholder to keep code compatible"""
    def __init__(self):
        pass

    def step(self, *args, **kwargs):
        pass


def get_kwargs(args, key):
    args_dict = vars(args).copy()
    if key + '_class' not in args_dict:
        return None, None
    clazz = args_dict[key + '_class']
    del args_dict[key + '_class']

    kwargs = {}
    for k, v in args_dict.items():
        if k.startswith(key):
            left, right = k.split('_', 1)
            if left == key:
                kwargs[right] = v
    return clazz, kwargs


def get_optim(args, parameters):
    """
    Reads the configuration and constructs a scheduler and optimizer
    :param args: the configuration Namespace
    :param parameters: model.parameters()
    :return: optimizer, scheduler
    if scheduler not specified a placeholder scheduler will be returned
    """
    optim_class, optim_kwargs = get_kwargs(args, 'optim')
    optim_class = getattr(torch.optim, optim_class)
    optim = optim_class(parameters, **optim_kwargs)
    scheduler_class, scheduler_kwargs = get_kwargs(args, 'scheduler')
    if scheduler_class is None:
        return optim, NullScheduler()
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)
    scheduler = scheduler_class(optim, **scheduler_kwargs)
    return optim, scheduler


def config(args=None):
    """
    Reads the command switches and creates a config
    Command line switches override config files
    :return: a Namespace of args
    """

    """ config """
    parser = argparse.ArgumentParser(description='configuration switches')
    parser.add_argument('-n', '--name', type=str, default=None)
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
    parser.add_argument('--model_init', type=str)

    """ hyper-parameters """
    parser.add_argument('--optim_class', type=str)
    parser.add_argument('--optim_lr', type=float)
    parser.add_argument('--scheduler_class', type=str)
    parser.add_argument('--batchsize', type=int)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_train_len', type=int)
    parser.add_argument('--dataset_test_len', type=int)
    parser.add_argument('--dataset_randomize', type=int)

    parser.add_argument('--data_aug_max_rotate', type=float)
    parser.add_argument('--data_aug_tps_variance', type=float)

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
        """
        Sets an argument if it's not already set in the args
        :param args: args namespace
        :param dict: a dict containing arguments to check
        :return:
        """
        for key, value in dict.items():
            if key in vars(args) and vars(args)[key] is None:
                vars(args)[key] = dict[key]
            elif key not in vars(args):
                vars(args)[key] = dict[key]
        return args

    """ 
    required due to https://github.com/yaml/pyyaml/issues/173
    pyyaml does not correctly parse scientific notation 
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    """ read the config file """
    if args.config is not None:
        with Path(args.config).open() as f:
            conf = yaml.load(f, Loader=loader)
            conf = flatten(conf)
            args = set_if_not_set(args, conf)

    """ args not set will be set to a default value """
    defaults = {
        'optim_class': 'Adam',
        'optim_lr': 1e-4,
        'checkpoint_freq': 1,
        'opt_level': 'O0',
        'display_kp_rows': 4,
        'display_freq': 5000,
        'model_init': 'kaiming_normal'
    }

    args = set_if_not_set(args, defaults)

    """ default to cuda:0 if device is not set"""
    if args.device is None:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    def counter():
        """
        counter to keep track of run id
        creates a file .run_id in the current directory which stores the most recent id
        """
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

    ''' if run_id not explicitly set, then guess it'''
    if args.run_id == -1:
        args.run_id = counter()

    return args
