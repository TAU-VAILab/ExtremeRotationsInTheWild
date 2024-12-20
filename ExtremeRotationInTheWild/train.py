import os
import yaml
import time
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from shutil import copy2
import csv


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Flow-based Point Cloud Generation Experiment')
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
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")
    

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')
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

    #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # Currently save dir and log_dir are the same
    config.log_name = "logs/%s_%s" % (cfg_file_name, run_time)
    config.save_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    config.log_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    os.makedirs(config.log_dir+'/config')
    copy2(args.config, config.log_dir+'/config')
    
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
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    if "test_loader_second" in loaders:
        test_loader_second = loaders['test_loader_second']
    if "train_loader_second" in loaders:
        train_loader_second = loaders['train_loader_second']
        
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    start_epoch = 0
    start_time = time.time()

    if args.resume:
        if args.pretrained is not None:
            start_epoch = trainer.resume(args.pretrained) + 1
        else:
            start_epoch = trainer.resume(cfg.resume.dir) + 1

    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    step = 0
    if cfg.data.stage_type in ["d_fov", "d_im"]:
        test_loader.dataset.random_ar.reset()
        
    # If test run, go through the validation loop first
    if args.test_run:
        trainer.save(epoch=-1, step=-1)
        val_info = trainer.validate(test_loader, epoch=-1)
        if "test_loader_second" in loaders:
            test_names = [test_loader.dataset.data_type,test_loader_second.dataset.data_type]
        if "train_loader_second" in loaders:
            trainers_names = [train_loader.dataset.data_type,train_loader_second.dataset.data_type+"_2"]

        if "test_loader_second" in loaders:
            val_info_second = trainer.validate(test_loader_second, epoch=-1)
            trainer.log_val(val_info, writer=writer, epoch=-1, step=step,val_info_second=val_info_second,tests_names =test_names)
        else:
            trainer.log_val(val_info, writer=writer, epoch=-1, step=step)
            
    # main training loop  
    second_iter = False
    for epoch in range(start_epoch, cfg.trainer.epochs):
        if "train_loader_second" in loaders:
            training_loader_iter = iter(train_loader_second)
        for bidx, data in enumerate(train_loader):
            if cfg.data.stage_type in ["d_fov", "d_im"]:
                train_loader.dataset.random_ar.reset()
            
            step = bidx + len(train_loader) * epoch + 1
            logs_info = trainer.update(data,data_type =train_loader.dataset.data_type)
            
            
            if "train_loader_second" in loaders:
                try:
                    data_second = next(training_loader_iter)
                except StopIteration:
                    training_loader_iter = iter(train_loader_second)
                    data_second = next(training_loader_iter)
                logs_info_second = trainer.update(data_second,data_type =train_loader_second.dataset.data_type)
                
            if step % int(cfg.viz.log_freq) == 0:
                duration = time.time() - start_time
                start_time = time.time()
                if "train_loader_second" in loaders:
                    if second_iter == True :
                        data_second_iter2 = next(training_loader_iter)
                        logs_info_second_iter2 = trainer.update(data_second_iter2,data_type =train_loader_second.dataset.data_type)
                        print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss1 %2.5f  Loss2 %2.5f Loss2_send_iter %2.5f"
                          % (epoch, bidx, len(train_loader), duration,
                             logs_info['loss'],logs_info_second['loss'],logs_info_second_iter2['loss']))
                    else:
                        print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss1 %2.5f  Loss2 %2.5f"
                          % (epoch, bidx, len(train_loader), duration,
                             logs_info['loss'],logs_info_second['loss']))
                else:
                    print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f"
                          % (epoch, bidx, len(train_loader), duration,
                             logs_info['loss']))
                visualize = step % int(cfg.viz.viz_freq) == 0
                if "train_loader_second" in loaders:
                    trainer.log_train(logs_info, data, writer=writer, epoch=epoch, step=step, 
                                      visualize=visualize,train_info_second=logs_info_second,trainers_names = trainers_names)
                else:
                    trainer.log_train(logs_info, data, writer=writer, epoch=epoch, step=step, visualize=visualize)
                
    
            if (step + 1) % int(cfg.viz.val_freq) == 0:
                val_info = trainer.validate(test_loader, epoch=epoch)
                trainer.log_val(val_info, writer=writer, epoch=epoch, step=step)
        trainer.save(epoch=epoch, step=step)
        val_info = trainer.validate(test_loader, epoch=epoch)
        
        if "test_loader_second" in loaders:
            val_info_second = trainer.validate(test_loader_second, epoch=epoch)
            trainer.log_val(val_info, writer=writer, epoch=epoch, step=step,val_info_second=val_info_second,tests_names = test_names)
        else:
            trainer.log_val(val_info, writer=writer, epoch=epoch, step=step)

        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, writer=writer)
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
