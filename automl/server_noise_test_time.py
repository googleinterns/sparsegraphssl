"""Runs experiments with noisy selections at test time."""
from automl.full_eval import start
from automl.full_eval import get_default_parser

if __name__ == '__main__':
  parser = get_default_parser()
  args = parser.parse_args()
  args.shared_directory = '/mnt/datasets/idea/noise_test_time/heterophily/'
  args.automl_worker = 'worker_noise_test_time'
  start(args)
