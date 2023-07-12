import importlib
import sys
import os

sys.setrecursionlimit(10000)
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import random
import torch.multiprocessing as mp
from networks.managers.trainer import Trainer

def main_worker(gpu, cfg, enable_amp=True):
    trainer = Trainer(rank=gpu, cfg=cfg, enable_amp=enable_amp)  # Initiate a training manager
    trainer.sequential_training()  # Start Training

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train VOS")
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    parser.add_argument('--stage', type=str, default='visha')
    parser.add_argument('--exp_name', type=str, default='lstn')  # 使用aot，非deaot
    parser.add_argument('--model', type=str, default='lstnt', choices=["lstnt", "lstns", "lstnb"])
    
    parser.add_argument('--gpu_num', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    args = parser.parse_args()

    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    cfg.TRAIN_GPUS = args.gpu_num if args.gpu_num > 0 else cfg.TRAIN_GPUS
    cfg.TRAIN_BATCH_SIZE = args.batch_size if args.batch_size > 0 else cfg.TRAIN_BATCH_SIZE
    cfg.DIST_URL = 'tcp://127.0.0.1:123' + str(random.randint(0, 9)) + str(random.randint(0, 9))

    if cfg.TRAIN_GPUS > 1:
        mp.spawn(main_worker, nprocs=cfg.TRAIN_GPUS, args=(cfg, args.amp))
    else:
        cfg.TRAIN_GPUS = 1
        main_worker(0, cfg, args.amp)

if __name__ == '__main__':
    main()
