# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse  # For parsing command-line arguments
import multiprocessing as mp  # For parallel processing across multiple GPUs
import pprint  # For pretty-printing dictionaries (e.g., config)
import yaml  # For loading YAML configuration files
import wandb  # For logging and visualization
import torch.distributed as dist
from app.scaffold import main as app_main  # Entry point for the pretraining pipeline
from src.utils.distributed import init_distributed  # Initializes distributed GPU communication
import os
# Argument parser for CLI inputs
parser = argparse.ArgumentParser()

parser.add_argument(
    '--fname', type=str,  # Path to YAML config file
    help='name of config file to load',
    default='configs.yaml')

parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],  # List of CUDA devices to use for training
    help='which devices to use on local machine')

parser.add_argument(  # NEW
    '--plotter', type=str, 
    default='wandb',  # Options: csv, wandb, tensorboard
    help='what plotter to use :csv or wandb or tensorboard')


parser.add_argument(
    '--debug' , type=lambda x: x.lower() == 'true',
    default=False,
    help='whether to run validation or not (True/False)'
)

parser.add_argument(
    '--save_mask' , type=lambda x: x.lower() == 'true',
    default=False,
    help='whether to save masks (True/False)'
)

def process_main(rank, fname, world_size, devices, plotter,debug,save_mask): #def process_main(rank, fname, world_size, devices):
    import os
    # import pdb; pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])  # Assign CUDA device per process

    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)  # Custom logger setup
    if rank == 0:
        logger.setLevel(logging.INFO)  # Master process logs info
    else:
        logger.setLevel(logging.ERROR)  # Worker processes suppress logs

    logger.info(f'called-params {fname}')  # Log path to config file

    # Load config
    params = None
    with open(fname, 'r') as y_file:  # Open YAML file
        params = yaml.load(y_file, Loader=yaml.FullLoader)  # Parse YAML to dict
        params['plotter'] = plotter  ## New # Inject CLI option into config
        logger.info('loaded params...')  # Log successful load

    # Log config (only by rank 0)
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)  # Pretty-print config
        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')  # Set path to save config copy
        with open(dump, 'w') as f:
            yaml.dump(params, f)  # Save loaded config to disk

    ###New by Hasitha
    if rank == 0 and plotter == 'wandb':
        wandb.init(
            entity="voxel-jepa",
            project=params['logging'].get('project', 'voxel-jepa-pretraining'),
            config=params,
            name=params['logging'].get('run_name', 'voxel-jepa') 
        )
    

    # # Init distributed (access to comm between GPUS on same machine)
    # world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))  # Setup DDP environment
     # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logger.info(f'Running... (rank: {rank}/{world_size})')  # Log DDP startup info
    
    # Launch the app with loaded config
    app_main(params['app'], args=params, debug=debug, save_mask=save_mask)  # Start pretraining via app.scaffold.main


if __name__ == '__main__':
    args = parser.parse_args() 
    # import pdb; pdb.set_trace()
    num_gpus = len(args.devices)  
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(params['port'])
    mp.set_start_method('spawn')  
    process_main(0,args.fname, num_gpus, args.devices, args.plotter, args.debug, args.save_mask)
