"""Provides random search experiments with ppi-dataset"""
from automl.full_eval import start
from automl.full_eval import get_default_parser

if __name__ == '__main__':
  parser = get_default_parser()
  args = parser.parse_args()
  args.shared_directory = '/mnt/datasets/idea/final/ppi_2/LP'
  args.min_budget = 1000
  args.max_budget = 1000
  args.automl_worker = 'worker_full_eval_ppi'
  start(args)
