"""Meaures the attention spent on server."""
from automl.full_eval import start
from automl.full_eval import get_default_parser

if __name__ == '__main__':
  parser = get_default_parser()
  args = parser.parse_args()
  args.shared_directory = '/mnt/datasets/idea/noise_test_time/shortest_distance/'
  args.automl_worker = 'worker_shortest_distance'
  start(args)
