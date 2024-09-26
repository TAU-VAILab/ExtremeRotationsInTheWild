import os
import yaml
import time
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
from shutil import copy2
from pprint import pprint
from tensorboardX import SummaryWriter
import re
import sys
import csv





def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")
    parser.add_argument('--val_angle', default=False, action='store_true',
                        help="Evaluate yaw and pitch error")
    parser.add_argument('--save', default=False,action='store_true',
                        help='to save bad images.')
    print(sys.argv)
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # Currently save dir and log_dir are the same
    config.log_name = "val_logs/%s_val_%s" % (cfg_file_name, run_time)
    config.save_dir = "val_logs/%s_val_%s" % (cfg_file_name, run_time)
    config.log_dir = "val_logs/%s_val_%s" % (cfg_file_name, run_time)
    os.makedirs(config.log_dir + '/config')
    copy2(args.config, config.log_dir + '/config')
    
    tags_path = os.path.join(config.log_dir+'/config', "meta_tags.csv")
    with open(tags_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in vars(args).items():
            writer.writerow(row)
            
    return args, config


def main_worker(gpu, ngpus_per_node, cfg, args):
    # basic setup
    cudnn.benchmark = True
    writer = SummaryWriter(logdir=cfg.log_name)

    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data)
    test_loader = loaders['test_loader']
    #visualize_matches_test(test_loader)
    #visualize_pair_test(test_loader)
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)
    trainer.resume(args.pretrained, test=True)
    val_info = trainer.validate(test_loader, epoch=-1, val_angle=args.val_angle,save_pictures=args.save)
    #val_info = trainer.validate(test_loader, epoch=-1, val_angle=args.val_angle)
    #trainer.log_val(val_info, writer=writer, step=-1)
    if "test_loader_second" in loaders:
         test_loader_second = loaders['test_loader_second']
         val_info_second = trainer.validate(test_loader_second, epoch=-1, val_angle=args.val_angle)
         test_names = [test_loader.dataset.data_type,test_loader_second.dataset.data_type]
         trainer.log_val(val_info, writer=writer, step=-1,val_info_second=val_info_second,tests_names = test_names)
    else:
        trainer.log_val(val_info, writer=writer, step=-1)
    print("Test done:")
    writer.close()


def main():
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, cfg, args)


if __name__ == '__main__':
    main()
