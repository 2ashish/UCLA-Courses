
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

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



import torch
from torch.autograd import Variable
from functools import reduce
import operator

count_ops = 0
count_params = 0
pruning_ratio = 0
strat = 'Structured'

def params_to_string(num_params, units=None, precision=2):
    """Convert parameter number into a string.
    Args:
        num_params (float): Parameter number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'M',
            'K' and ''. If set to None, it will automatically choose the most
            suitable unit for Parameter number. Default: None.
        precision (int): Digit number after the decimal point. Default: 2.
    Returns:
        str: The converted parameter number with units.
    Examples:
        >>> params_to_string(1e9)
        '1000.0 M'
        >>> params_to_string(2e5)
        '200.0 k'
        >>> params_to_string(3e-9)
        '3e-09'
    """
    if units is None:
        if num_params // 10**9 > 0:
            return str(round(num_params / 10**9, precision)) + ' G'
        elif num_params // 10**6 > 0:
            return str(round(num_params / 10**6, precision)) + ' M'
        elif num_params // 10**3:
            return str(round(num_params / 10**3, precision)) + ' K'
        else:
            return str(num_params)
    else:
        if units == 'G':
            return str(round(num_params / 10.**9, precision)) + ' ' + units
        elif units == 'M':
            return str(round(num_params / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(num_params / 10.**3, precision)) + ' ' + units
        else:
            return str(num_params)


def get_num_gen(gen):
    return sum(1 for x in gen)

def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_layer_info(layer):
    layer_str = str(layer)
    # print(layer_str)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model, is_conv=True):
    if is_conv:
        total=0.
        for idx, param in enumerate(model.parameters()):
            assert idx<2
            f = param.size()[0]
            pruned_num = int(pruning_ratio * f)
            if len(param.size())>1:
                c=param.size()[1]
                if hasattr(model,'last_prune_num'):
                    last_prune_num=model.last_prune_num
                    total += (f - pruned_num) * (c-last_prune_num) * param.numel() / f / c
                else:
                    total += (f - pruned_num) * param.numel() / f
            else:
                total += (f - pruned_num) * param.numel() / f
        return total
    else:
        return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

### The input batch size should be 1 to call this function
def measure_layer(layer, x, print_name):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        pruned_num = int(pruning_ratio * layer.out_channels)
        pruned_last_num = max(3,int(pruning_ratio * layer.in_channels))
        
        if strat == 'Structured':
            delta_ops = (layer.in_channels-pruned_last_num) * (layer.out_channels - pruned_num) * layer.kernel_size[0] * \
                        layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        elif strat == 'Unstructured':
            in_connect = (layer.in_channels*layer.kernel_size[0] *layer.kernel_size[1])
            in_connect = in_connect - int(in_connect*pruning_ratio)
            out_connect = (layer.out_channels*out_h *out_w)
            out_connect = out_connect - int(out_connect*pruning_ratio)
            delta_ops = in_connect*out_connect / layer.groups * multi_add
        else:
            raise TypeError('unknown pruning strategy: %s' % strat)

        delta_ops_ori = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add

        delta_params = get_layer_param(layer)

        #if print_name:
        #    print(type_name, pruning_ratio, '| input:',x.size(),'| weight:',[layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1]],
        #          '| params:', delta_params, '| flops:', delta_ops_ori)
        #else:
        #    print(pruning_ratio, [layer.out_channels,layer.in_channels,layer.kernel_size[0],layer.kernel_size[1]],
        #          'params:',delta_params, ' flops:',delta_ops_ori)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer, is_conv=False)

        print('linear:',layer, delta_ops, delta_params)

    elif type_name in ['DenseBasicBlock', 'ResBasicBlock']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Inception']:
        measure_layer(layer.conv1, x)

    elif type_name in ['DenseBottleneck', 'SparseDenseBottleneck']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Transition', 'SparseTransition']:
        measure_layer(layer.conv1, x)

    elif type_name in ['CharbonnierLoss','Upsample','LeakyReLU', 'ReLU', 'BatchNorm1d','BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'AdaptiveAvgPool2d', 'AvgPool2d', 'MaxPool2d', 'Mask', 'channel_selection', 'LambdaLayer', 'Sequential']:
        return 
    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return

def measure_model(model,data,print_name=False,strat='Structured'):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    
    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model, print_name):
        for child in model.children():
            if should_measure(child):
                #print(get_layer_info(child))
                def new_forward(m):
                    def lambda_forward(x,y=None):
                        measure_layer(m, x, print_name)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child, print_name)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model, print_name)
    model.forward(data)
    restore_forward(model)

    return params_to_string(count_ops), params_to_string(count_params)

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
    global pruning_ratio,strat
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
    input_shape = (1,1,3,128,128)
    data = Variable(torch.zeros(input_shape)).to(torch.device("cuda"))
    model = load_model(args)
    
    flops, params = measure_model(model, data,print_name=True)
    split_line = '=' * 30
    s = ''
    s+= f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}\n'
    #print(f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\n'
    #    f'Flops: {flops}\nParams: {params}\n{split_line}\n')
    
    strat = 'Unstructured'
    for i in range(1,20,1):
        pruning_ratio = i*0.05
        args.checkpoint = osp.join(args.work_dir, 'basicvsr_reds4_unstruct_'+str(int(pruning_ratio*100))+'.pth')
        model = load_model(args)
        flops, params = measure_model(model, data,print_name=True)
        s+= f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}\n'
        #print(f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\n'
        #    f'Flops: {flops}\nParams: {params}\n{split_line}\n')
    
    strat='Structured'
    for i in range(1,20,1):
        pruning_ratio = i*0.05
        args.checkpoint = osp.join(args.work_dir, 'basicvsr_reds4_struct_'+str(int(pruning_ratio*100))+'.pth')
        model = load_model(args)
        flops, params = measure_model(model, data,print_name=True)
        s+= f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}\n'
        #print(f'{split_line}\n{args.checkpoint}\nInput shape: {input_shape}\n'
        #    f'Flops: {flops}\nParams: {params}\n{split_line}\n')
    
    print(s)
    
    
if __name__ == '__main__':
    main()
