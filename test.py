import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import sys
sys.path.append('./lstn')

from lstn.tools.eval import get_parser
from lstn.networks.managers.evaluator_visha import EvaluatorVisha


if __name__ == '__main__':
    cfg = get_parser()
    evaluator = EvaluatorVisha(cfg)
    evaluator.evaluating()