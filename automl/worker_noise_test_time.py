"""A worker to be used with automl.

This worker provides the standard settings to run models automatically.

Attributes:
    STANDARD_CONFIG (dict): Some configs to be used in every experiments.
"""
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from automl.worker_noise_selection import PyTorchWorker as Worker

STANDARD_CONFIG = {
    'patience': 100,
    'repeated_runs': 0,
    'epochs': 1000,
    'pseudo_label_mode': 'final_only',
    'label_prop_steps': 1,
    'use_adj_matrix': False,
    'vis_att': False,
    'att_analysis': False,
    'noise_robustness': True,
    'noise_type': 'heterophily',
}


class PyTorchWorker(Worker):
  """Worker class for citations datasets.

  Attributes:
      repeat_per_run (int): how many times to run/config.
      standard_config (dict): which configs should be used.
  """

  def __init__(self, **kwargs):
    """Inits the worker.

    Args:
        **kwargs: keyword-arguments to init parent classes.
    """
    super().__init__(**kwargs)
    self.repeat_per_run = 1
    self.standard_config = STANDARD_CONFIG

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

  worker = PyTorchWorker(run_id='0', gpu=-1)
  new_configspace = worker.get_configspace()
  config = new_configspace.sample_configuration().get_dictionary()
  print(config)
  BUDGET = 1
  config['model'] = 'GAT'
  res = worker.compute(config=config, budget=BUDGET, working_directory='.')
  print(res)
