import importlib
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('./lstn')

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Eval ShadowSAM")
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    parser.add_argument('--stage', type=str, default='visha')
    parser.add_argument('--exp_name', type=str, default='lstn')
    parser.add_argument('--model', type=str, default='lstnt', choices=["lstnt", "lstns", "lstnb"])

    parser.add_argument('--start_step', type=int, default=10000)
    # parser.add_argument('--ckpt_step', type=int, default=-1)
    parser.add_argument('--ckpt_path', type=str, default='./lstn/checkpoints/lstnt')

    parser.add_argument('--datapath', type=str, default='')

    parser.add_argument('--ema', action='store_true')
    parser.set_defaults(ema=False)
    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)

    args = parser.parse_args()

    print(args.start_step)

    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    cfg.TEST_EMA = args.ema
    cfg.TEST_CKPT_STEP = args.start_step
        
    cfg.TEST_CKPT_PATH = args.ckpt_path
    cfg.TEST_DATASET_PATH = args.datapath if args.datapath != '' else cfg.TEST_DATASET_PATH
    cfg.TEST_MAX_LONG_EDGE = args.max_resolution * 800. / 480.

    return cfg