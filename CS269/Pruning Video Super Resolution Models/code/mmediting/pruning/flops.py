# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
import torch.nn.utils.prune as prune
from mmcv import Config, DictAction
from mmcv.runner import init_dist,load_checkpoint,save_checkpoint
from mmcv.cnn.utils import get_model_complexity_info

from mmedit import __version__
from mmedit.apis import init_random_seed, set_random_seed, train_model
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Prune basic vsr')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', default='prune_ckpts', help='the dir to save checkpoints')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    args = parser.parse_args()
    return args

def load_model(args):
    # model settings
    model_settings = dict(
        type='BasicVSR',
        generator=dict(
            type='BasicVSRNet',
            mid_channels=64,
            num_blocks=30,
            spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
            'basicvsr/spynet_20210409-c6c1bd09.pth'),
        pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
    # model training and testing settings
    train_cfg = dict(fix_iter=5000)
    test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)
    
    model = build_model(
        model_settings, train_cfg=train_cfg, test_cfg=test_cfg)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda()
    model.eval()
    
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported '
            f'with {model.__class__.__name__}')
            
    return model
        
def main():
    args = parse_args()
    
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    # set random seeds
    seed = init_random_seed(args.seed)
    input_shape = (1,3,128,128)
    model = load_model(args)
    
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    s = ''
    s+= f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}\n'
    print(f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\n'
        f'Flops: {flops}\nParams: {params}\n{split_line}\n')
    #print(s)
            
    for i in range(1,20,1):
        pruning_ratio = i*0.05
        args.checkpoint = osp.join(args.work_dir, 'basicvsr_reds4_unstruct_'+str(int(pruning_ratio*100))+'.pth')
        model = load_model(args)
        flops, params = get_model_complexity_info(model, input_shape)
        s+= f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}\n'
        print(f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\n'
            f'Flops: {flops}\nParams: {params}\n{split_line}\n')
        
    print(s)
    
    
if __name__ == '__main__':
    main()
