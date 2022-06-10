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

from mmedit import __version__
from mmedit.apis import init_random_seed, set_random_seed, train_model
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Prune basic vsr')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', default='prune_ckpts_meta', help='the dir to save checkpoints')
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
    ckp = load_checkpoint(model, args.checkpoint, map_location='cpu')
    return model,ckp['meta']
    
def unstruct_pruning(args,model,meta,pruning_ratio):
    #l1 unstructured pruning
    for name, module in model.named_modules():
        # prune some % of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
        else:
            pass
    
    save_checkpoint(model,osp.join(args.work_dir, 'basicvsr_reds4_unstruct_'+str(int(pruning_ratio*100))+'.pth'),meta=meta)
    #print("ckpt pruned and saved")
    return
    
def struct_pruning(args,model,meta,pruning_ratio):
    #l1 structured pruning
    for name, module in model.named_modules():
        # prune some % of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight',n=1,dim=0, amount=pruning_ratio)
            prune.remove(module, 'weight')
        else:
            pass
    save_checkpoint(model,osp.join(args.work_dir, 'basicvsr_reds4_struct_'+str(int(pruning_ratio*100))+'.pth'),meta=meta)
    #print("ckpt pruned and saved")
    return
    
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
    
    for i in range(1,15,1):
        pruning_ratio = i*0.05
        model,meta = load_model(args)
        unstruct_pruning(args,model,meta,pruning_ratio)
        print("Pruned and saved ckpt for ratio: ",pruning_ratio)
        
    for i in range(1,10,1):
        pruning_ratio = i*0.05
        model,meta = load_model(args)
        struct_pruning(args,model,meta,pruning_ratio)
        print("Pruned and saved ckpt for ratio: ",pruning_ratio)
    
    
if __name__ == '__main__':
    main()
