"""Tests different model architectures."""
import pytest
from automl.worker_full_eval import PyTorchWorker
from run_scripts.gat_train import get_args
from run_scripts.gat_train import main as train_model


@pytest.mark.gcn_simple
def test_gcn_simple():
  """Tests gcn model."""

  args = get_args()
  config = args.__dict__
  config['model'] = 'gcn'
  config['main_loss'] = 'entropy_only'
  config['in_drop'] = 0
  config['repeated_runs'] = 0
  config['lr'] = 0.01
  config['patience'] = 100
  config['att_last_layer'] = True
  config['gpu'] = -1
  config['num_layers'] = 2

  config['data'] = 'citeseer'
  config['labeling_rate'] = 0.01

  budget = 1

  config['epochs'] = budget
  print(config)
  res = train_model(args)
  print(res)


@pytest.mark.gcn_full_eval
def test_gcn_full_eval():
  """Tests gcn model with high labeling rates."""
  seeds_array = [5006, 690, 42]
  worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

  cs = worker.get_configspace()
  config = cs.sample_configuration().get_dictionary()

  config['model'] = 'gcn'
  config['main_loss'] = 'entropy_only'
  config['in_drop'] = 0
  config['repeated_runs'] = 0
  config['lr'] = 0.001
  config['patience'] = 100
  config['att_last_layer'] = True

  config['data'] = 'cora'
  config['labeling_rate'] = 1

  budget = 10

  print(config)
  res = worker.compute(config=config, budget=budget, working_directory='.')
  assert res, 'something is wrong'
  print(res)


@pytest.mark.graphsage_simple
def test_graphsage_simple():
  """Tests graphsage model."""
  args = get_args()
  config = args.__dict__
  config['model'] = 'graphsage'
  config['main_loss'] = 'entropy_only'
  config['in_drop'] = 0
  config['repeated_runs'] = 0
  config['lr'] = 0.001
  config['patience'] = 100
  config['att_last_layer'] = True
  config['gpu'] = -1

  config['data'] = 'cora'
  config['labeling_rate'] = 1

  budget = 1

  config['epochs'] = budget
  print(config)
  res = train_model(args)
  print(res)


@pytest.mark.graphsage_full_eval
def test_graphsage_full_eval():
  """Tests graphsage model with high labeling rates."""
  seeds_array = [5006, 690, 42]
  worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

  cs = worker.get_configspace()
  config = cs.sample_configuration().get_dictionary()

  config['model'] = 'graphsage'
  config['main_loss'] = 'entropy_only'
  config['in_drop'] = 0
  config['repeated_runs'] = 0
  config['lr'] = 0.001
  config['patience'] = 100
  config['att_last_layer'] = True

  config['data'] = 'cora'
  config['labeling_rate'] = 1

  budget = 10

  print(config)
  res = worker.compute(config=config, budget=budget, working_directory='.')
  assert res, 'something is wrong'
  print(res)
