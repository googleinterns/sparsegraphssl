"""A worker to be used with automl.

This worker provides the standard settings to run models automatically.

Attributes:
    STANDARD_CONFIG (dict): Some configs to be used in every experiments.
"""
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from automl.automl_utils import add_graphsage_config
from automl.automl_utils import add_gat_config
from automl.automl_utils import add_dropout_ignore_gat
from hpbandster.core.worker import Worker
from run_scripts.gat_train_ppi import main as main_ppi
from run_scripts.gat_train_ppi import get_args as get_args_ppi
from run_scripts.gat_train import get_args as get_args_ctgr

from run_scripts.gat_train import main as main_ctgr
import gc
import torch

STANDARD_CONFIG = {'budget': 1000}


class PyTorchWorker(Worker):
  """Worker class for citations datasets.

  Attributes:
      config (dict): hyperparameters to be used
      gpu (int): which gpu to be used.
      repeat_per_run (int): how many times to run/config.
      seeds (int): single seed to be used.
      seeds_array (list): which seeds to be used in which order.
  """

  def __init__(self, args=None, gpu=-1, seeds_array=None, **kwargs):
    """Inits the worker.

    Args:
        args (None, optional): all configs
        gpu (int, optional): which gpu to be changed
        seeds_array (None, optional): Description
        **kwargs: keyword-arguments not needed.
    """
    super().__init__(**kwargs)
    self.repeat_per_run = 5
    self.gpu = gpu
    self.seeds_array = seeds_array
    self.config = args
    gc.collect()
    torch.cuda.empty_cache()

  def add_config(self, config, budget):
    """Add and updates all configs.

    Args:
        config (dict): hyperparameters to be used.
        budget (int): epochs for training.

    Returns:
        new_config: updated hyperparameters set.

    Raises:
        NotImplementedError: Not supported data error.
    """
    if config['data'] in [
        'ppi', 'ppi_subsampled', 'elliptic', 'ppi_subgraph_ssl'
    ]:
      args_config = get_args_ppi().parse_args([])
    elif config['data'] in ['cora', 'citeseer', 'pubmed'] \
      or config['data'].startswith('yelp'):
      args_config = get_args_ctgr().parse_args([])
    else:
      raise ValueError('Data not supported: %s' % config['data'])
    args_config_dict = args_config.__dict__
    args_config_dict.update(STANDARD_CONFIG)
    args_config_dict.update(config)

    args_config_dict['gpu'] = self.gpu
    args_config_dict['epochs'] = int(budget)
    if 'gcn_parameters' in args_config_dict:
      args_config_dict['gcn_parameters'].update(config)
    if 'sparse_gat_parameters' in args_config_dict:
      args_config_dict['sparse_gat_parameters'].update(config)
    return args_config

  def subcompute(self, config, budget, *args, **kwargs):
    """Runs a single experiment.

    This function runs one single experiment based on one config.

    Args:
        config (dict): hyperparameters to be used.
        budget (int): epochs for training.
        *args: positional arguments not needed.
        **kwargs: keyword-arguments not needed.

    Returns:
        best_results: dictionary of resutls.

    Raises:
        NotImplementedError: Data not supported error.
    """
    if config['data'] in ['cora', 'citeseer', 'pubmed']:
      self.config = self.add_config(config, budget)
      best_test_perf, results = main_ctgr(self.config)
    elif config['data'] in ['ppi', 'ppi_subsampled', 'ppi_subgraph_ssl']:
      self.config = self.add_config(config, budget)
      best_test_perf, results = main_ppi(self.config)
    elif config['data'].startswith('yelp'):
      self.config = self.add_config(config, budget)
      best_test_perf, results = main_ctgr(self.config)
    else:
      raise ValueError('Data not supported: %s' % config['data'])
    gc.collect()
    torch.cuda.empty_cache()

    best_epoch, best_test_perf, best_test_perf_f1, best_valid_perf, best_valid_perf_f1 = results
    best_results = ({
        'loss': 1 - best_valid_perf,  # remember: HpBandSter always minimizes!
        'info': {
            'best_epoch': best_epoch,
            'best_test_perf': best_test_perf,
            'best_test_perf_f1': best_test_perf_f1,
            'best_valid_perf': best_valid_perf,
            'best_valid_perf_f1': best_valid_perf_f1,
        }
    })
    return best_results

  def compute(self, config, budget, *args, **kwargs):
    """Runs experiments muliple times.

    This function runs experiment muliple times and gather results.

    Args:
        config (dict): hyperparameters to be used
        budget (int): epochs for training.
        *args: positional arguments not needed.
        **kwargs: keyword-arguments not needed.

    Returns:
        best_results: dictionary of resutls.
    """
    if 'gat' in config['model'] and config['data'].startswith('yelp'):
      config['num_hidden'] = 64

    self.seeds = [
        config['repeated_runs'],
    ] * self.repeat_per_run

    if 'lambda_sparsemax' in config:
      config['lambda_sparsemax'] = -(config['lambda_sparsemax'] - 1)
    if config['data'] == 'pubmed':
      config['num_out_heads'] = 8
      config['weight-decay'] = 0.001
    elif config['data'] in ['cora', 'citeseer']:
      config['num_out_heads'] = 1
      config['weight-decay'] = 5e-4
    results = np.zeros((len(self.seeds), 5))

    for i, seed in enumerate(self.seeds):

      config['repeated_runs'] = seed

      current_results = self.subcompute(config, budget, *args, **kwargs)
      info = current_results['info']
      best_epoch = info['best_epoch']
      best_test_perf = info['best_test_perf']
      best_test_perf_f1 = info['best_test_perf_f1']
      best_valid_perf = info['best_valid_perf']
      best_valid_perf_f1 = info['best_valid_perf_f1']
      added_info = np.array([
          best_epoch, best_test_perf, best_test_perf_f1, best_valid_perf,
          best_valid_perf_f1
      ])
      results[i, :] = np.array(added_info)
    assert (len(results)) == self.repeat_per_run
    best_epoch = results[:, 0].mean()
    best_test_perf = results[:, 1].mean()
    best_test_perf_std = results[:, 1].std()
    best_test_perf_sec = results[:, 2].mean()
    best_valid_perf = results[:, 3].mean()
    best_valid_perf_std = results[:, 3].std()
    best_valid_perf_sec = results[:, 4].mean()

    best_results = ({
        'loss': 1 - best_valid_perf,
        'info': {
            'best_epoch': best_epoch,
            'best_test_perf': best_test_perf,
            'best_test_perf_f1': best_test_perf_sec,
            'best_test_perf_std': best_test_perf_std,
            'best_valid_perf': best_valid_perf,
            'best_valid_perf_f1': best_valid_perf_sec,
            'best_valid_perf_std': best_valid_perf_std,
        }
    })

    all_results = {'config': config, 'results': best_results}
    self.append_pickle('tmp_results.txt', all_results)
    return best_results

  def append_pickle(self, cache_file, data):
    """Saves the results addtionally into some txt files.

    Args:
        cache_file (str): where to save the results.
        data (dict): results of runs.
    """
    import json
    with open(cache_file, 'a') as f:
      json.dump(data, f)
      f.write('\n')

  @staticmethod
  def get_configspace():
    """Get configuration for a experiment run.

    This staticmethod provides the configuration space to perform sampling from.

    Returns:
        cs: Configspace with all hyperparameters.
    """
    configspace = CS.ConfigurationSpace()

    all_models_names = [
        'LabelPropGAT',
    ]
    model = CSH.CategoricalHyperparameter('model', all_models_names)
    datasets = ['citeseer', 'cora', 'pubmed']
    num_layers = CSH.CategoricalHyperparameter('num_layers', [1])
    data = CSH.CategoricalHyperparameter('data', datasets)

    repeated_runs = CSH.CategoricalHyperparameter('repeated_runs', range(5))
    lr = CSH.CategoricalHyperparameter('lr', [.005])

    patience = CSH.CategoricalHyperparameter('patience', [100])

    configspace.add_hyperparameters(
        [model, data, num_layers, patience, lr, repeated_runs])

    samples_per_class = CSH.CategoricalHyperparameter('samples_per_class',
                                                      [1, 2, 5, 10, 20])
    configspace.add_hyperparameters([samples_per_class])
    if 'GAT' in all_models_names:
      configspace = add_gat_config(configspace, model)
    if 'GraphSAGE' in all_models_names:
      configspace = add_graphsage_config(configspace, model)

    if not (len(all_models_names) == 1 and
            all_models_names[0] in ['GAT', 'UnsupSparseGAT', 'SparseLossGAT']):
      configspace = add_dropout_ignore_gat(configspace, model, all_models_names)

    pseudo_label_mode = CSH.CategoricalHyperparameter('pseudo_label_mode',
                                                      ['final_only'])
    label_prop_steps = CSH.CategoricalHyperparameter('label_prop_steps',
                                                     range(5))
    configspace.add_hyperparameters([pseudo_label_mode, label_prop_steps])

    pooling_residual = CSH.CategoricalHyperparameter('pooling_residual',
                                                     [False, True])
    use_adj_matrix = CSH.CategoricalHyperparameter('use_adj_matrix',
                                                   [False, True])
    configspace.add_hyperparameters([pooling_residual, use_adj_matrix])
    return configspace


if __name__ == '__main__':
  """Example usage."""

  seeds_array = [5006, 690, 42]
  worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

  new_configspace = worker.get_configspace()
  config = new_configspace.sample_configuration().get_dictionary()
  BUDGET = 10
  config['samples_per_class'] = 1
  config['label_prop_steps'] = 3
  config['in_drop'] = 0
  config['attn_drop'] = 0
  config['data'] = 'citeseer'
  config['repeated_runs'] = 0
  config['model'] = 'LabelPropSparseGAT'
  config['patience'] = 100
  config['pooling_residual'] = True
  config['use_adj_matrix'] = False

  worker.repeat_per_run = 1
  print(config)
  res = worker.compute(config=config, budget=BUDGET, working_directory='.')
  print(res)
