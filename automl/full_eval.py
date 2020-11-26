"""Handles automl for starting experiments."""

import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.result import Result
from ext_hpband.full_search import FullSearch as BOHB
import copy
import importlib


def start(args):
  """Summary

  Args:
      args (namespace): all arguments for this file.
  """
  worker_module = importlib.import_module('%s.%s' %
                                          ('automl', args.automl_worker))
  worker = getattr(worker_module, 'PyTorchWorker')
  args.seeds_array = [5006, 690, 42]

  host = '127.0.0.1'
  if args.worker:
    import time
    time.sleep(
        1
    )  # short artificial delay to make sure the nameserver is already running
    w = worker(
        gpu=args.gpu,
        seeds_array=args.seeds_array,
        run_id=args.run_id,
        host=host,
        timeout=120,
        args=args)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)

  if os.path.exists('tmp_results.txt'):
    os.remove('tmp_results.txt')

  # This example shows how to log live results. This is most useful
  # for really long runs, where intermediate results could already be
  # interesting. The core.result submodule contains the functionality to
  # read the two generated files (results.json and configs.json) and
  # create a Result object.
  result_logger = hpres.json_result_logger(
      directory=args.shared_directory, overwrite=True)

  # Start a nameserver:
  nameserver = hpns.NameServer(
      run_id=args.run_id,
      host=host,
      port=None,
      working_directory=args.shared_directory)
  print('##############################################')
  print('#### STARTING SERVER ######')
  print('##############################################')
  ns_host, ns_port = nameserver.start()

  # Start local worker
  if os.path.exists(os.path.join(args.previous_run_dir, 'results.pkl')):
    previous_run = hpres.logged_results_to_HBS_result(args.previous_run_dir)
  else:
    previous_run = None
  # Run an optimizer
  bohb = BOHB(
      configspace=worker.get_configspace(),
      run_id=args.run_id,
      host=host,
      nameserver=ns_host,
      nameserver_port=ns_port,
      result_logger=result_logger,
      min_budget=args.min_budget,
      max_budget=args.max_budget,
      previous_result=previous_run,
  )

  try:
    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=1)
  except KeyboardInterrupt:
    print('Keyboard Interrupted server. Trying to save resutls anyway')
    for i in bohb.warmstart_iteration:
      i.fix_timestamps(bohb.time_ref)
    ws_data = [i.data for i in bohb.warmstart_iteration]
    res = Result([copy.deepcopy(i.data) for i in bohb.iterations] + ws_data,
                 bohb.config)

  # shutdown
  bohb.shutdown(shutdown_workers=True)
  nameserver.shutdown()
  # store results
  with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)


def get_default_parser():
  """Provides the default parser.

  This fucntion gives the default parser to be configured.

  Returns:
      parser: a parser to be used
  """
  parser = argparse.ArgumentParser(description='Tuning Params')
  parser.add_argument(
      '--min_budget',
      type=float,
      help='Minimum number of epochs for training.',
      default=400)
  parser.add_argument(
      '--max_budget',
      type=float,
      help='Maximum number of epochs for training.',
      default=400)
  parser.add_argument(
      '--n_iterations',
      type=int,
      help='Number of iterations performed by the optimizer',
      default=200)

  parser.add_argument(
      '--previous_run_dir',
      type=str,
      help='A directory that contains a config.json and results.json for the same configuration space.',
      default='/mnt/datasets/elliptic/')

  parser.add_argument(
      '--worker',
      help='Flag to turn this into a worker process',
      action='store_true')
  parser.add_argument(
      '--run_id',
      type=str,
      help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.',
      default='Elliptic_Study')
  parser.add_argument(
      '--shared_directory',
      type=str,
      help='A directory that is accessible for all processes, e.g. a NFS share.',
      default='/mnt/datasets/idea/sparse_gat/')
  parser.add_argument(
      '--gpu', type=int, help='Which gpu to be used. -1=Cpu', default=-1)

  return parser


if __name__ == '__main__':
  """Provides an example usage"""
  parser = get_default_parser()
  args = parser.parse_args()
  args.shared_directory = '/mnt/datasets/idea/sparsity_all/'
  args.min_budget = 2
  args.max_budget = 2
  start(args)
