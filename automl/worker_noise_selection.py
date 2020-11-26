"""A worker to be used with automl.

This worker provides the standard settings to run models automatically.

Attributes:
    STANDARD_CONFIG (dict): Some configs to be used in every experiments.
"""
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from automl.worker_full_eval import PyTorchWorker as Worker
from run_scripts.gat_train import main as main_ctgr
import pdb

STANDARD_CONFIG = {
    'patience': 100,
    'repeated_runs': 0,
    'epochs': 1000,
    'pseudo_label_mode': 'final_only',
    'label_prop_steps': 1,
    'use_adj_matrix': False,
    'vis_att': False,
    'att_analysis': False,
    'noise_robustness': False,
    'noise_type': 'heterophily',
}


class PyTorchWorker(Worker):
  """Worker class for citations datasets.

  Attributes:
      repeat_per_run (int): how many times to run/config.l
      standard_config (dict): which configs to be used.
  """

  def __init__(self, **kwargs):
    """Inits the worker.

    Args:
        **kwargs: keyword-arguments not needed.
    """
    super().__init__(**kwargs)
    self.repeat_per_run = 1
    self.standard_config = STANDARD_CONFIG

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
    if config['data'] == 'pubmed':
      config['num_out_heads'] = 8
      config['weight-decay'] = 0.001
    elif config['data'] in ['cora', 'citeseer']:
      config['num_out_heads'] = 1
      config['weight-decay'] = 5e-4

    config = self.add_config(config, budget)
    config.__dict__.update(self.standard_config)
    print(config)
    best_test_perf, results = main_ctgr(config)
    best_epoch, best_test_perf, best_valid_perf, att_on_noisy_edge = results

    best_results = ({
        'loss': 1 - best_test_perf,
        'info': {
            'best_epoch': best_epoch,
            'best_test_perf': best_test_perf,
            'best_valid_perf': best_valid_perf,
            'att_on_noisy_edge': att_on_noisy_edge,
        }
    })
    all_results = {'config': config.__dict__, 'results': best_results}
    self.append_pickle('tmp_results.txt', all_results)
    return best_results

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
        'LabelPropSparseGAT',
        'GAT',
        'SparseGAT',
    ]
    model = CSH.CategoricalHyperparameter('model', all_models_names)
    datasets = ['citeseer', 'cora', 'pubmed']
    data = CSH.CategoricalHyperparameter('data', datasets)

    configspace.add_hyperparameters([model, data])

    samples_per_class = CSH.CategoricalHyperparameter('samples_per_class',
                                                      [1, 2, 5, 10, 20])
    configspace.add_hyperparameters([samples_per_class])
    noise_injected_ratio = CSH.CategoricalHyperparameter(
        'noise_injected_ratio', [.05, .1, .2, .5, 1])
    configspace.add_hyperparameters([noise_injected_ratio])
    # add lambda_sparsemax -parameter for sparsemax-models
    models_lambda = ['SparseGAT', 'LabelPropSparseGAT']
    lambda_sparsemax = CSH.CategoricalHyperparameter('lambda_sparsemax',
                                                     [-1, .0, .5, .99])
    configspace.add_hyperparameters([lambda_sparsemax])
    cond_list = []
    for model_name in models_lambda:
      cond = CS.EqualsCondition(lambda_sparsemax, model, model_name)
      cond_list.append(cond)
    if len(cond_list) > 1:
      configspace.add_condition(CS.OrConjunction(*cond_list))
    else:
      configspace.add_condition(*cond_list)

    return configspace


if __name__ == '__main__':
  """Example usage."""

  seeds_array = [5006, 690, 42]
  worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

  new_configspace = worker.get_configspace()
  config = new_configspace.sample_configuration().get_dictionary()
  BUDGET = 10
  print(config)
  res = worker.compute(config=config, budget=BUDGET, working_directory='.')
  print(res)
