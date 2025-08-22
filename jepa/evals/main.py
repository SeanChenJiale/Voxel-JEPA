# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed

from evals.scaffold import main as eval_main
import torch.distributed as dist
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--plotter', type=str,
    default='wandb',
    help='what plotter to use: csv or wandb')
parser.add_argument(
    '--validation', type=lambda x: x.lower() == 'true',
    default=False,
    help='whether to run validation or not (True/False)'
)
parser.add_argument(
    '--debug' , type=lambda x: x.lower() == 'true',
    default=False,
    help='whether to run validation or not (True/False)'
)
parser.add_argument(
    '--grad_cam', type=lambda x: x.lower() == 'true',
    default=False,
    help='whether to run grad_cam or not (True/False)'
)

def process_main(rank, fname, world_size, devices, plotter, validation, debug, grad_cam):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')
    

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        # params['plotter'] = plotter  # Add plotter to config
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # # Init distributed (access to comm between GPUS on same machine) ORIGINAL
    # world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))

    dist.init_process_group("gloo", rank=rank, world_size=world_size) #NEW
    logger.info(f'Running... (rank: {rank}/{world_size})')
    
    ### New part by Hasitha
    if rank == 0 and plotter == 'wandb':
        import wandb
        wandb.init(
            entity="voxel-jepa",
            project=params['logging'].get('project', 'voxel-jepa-fine-tuning'),
            config=params,
            name=params['logging'].get('run_name', 'voxel-finetune-test')
        )
    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params,plotter=plotter, validation=validation, debug=debug, grad_cam=grad_cam)



if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(params['port'])
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.plotter, args.validation, args.debug, args.grad_cam)
        ).start()
