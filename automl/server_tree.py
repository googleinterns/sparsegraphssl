"""Runs experiments to generate explanitions tree."""
from automl.full_eval import start
from automl.full_eval import get_default_parser

if __name__ == '__main__':
  parser = get_default_parser()
  args = parser.parse_args()
  args.shared_directory = '/mnt/datasets/idea/tree'
  args.automl_worker = 'worker_tree'
  start(args)
