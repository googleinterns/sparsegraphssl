"""Provides random search experiments with citation datasets"""
from automl.full_eval import start
from automl.full_eval import get_default_parser

if __name__ == '__main__':
  parser = get_default_parser()
  args = parser.parse_args()
  args.shared_directory = '/mnt/datasets/idea/final/ctgr_8/SIGN+LP'
  args.min_budget = 1
  args.max_budget = 1
  args.automl_worker = 'worker_full_eval'
  start(args)
