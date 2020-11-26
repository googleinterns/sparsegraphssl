"""A worker to be used with automl.

This worker provides the standard settings to run models automatically.

"""
from automl.automl_utils import add_graphsage_config
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from automl.worker_full_eval import PyTorchWorker as EvalWorker


class PyTorchWorker(EvalWorker):
  """Worker class for ppi.

  Attributes:
      repeat_per_run (int): how many times to run/config.l
  """

  def __init__(self, args=None, **kwargs):
    """Summary

    Args:
        args (None, optional): Description
        **kwargs: Description
    """
    super().__init__(**kwargs)
    self.repeat_per_run = 3

  def compute(self, config, budget, *args, **kwargs):
    """Summary

    Args:
        config (dict): hyperparameters to be used.
        budget (int): epochs for training.
        seeds_array (None, optional): Description
        **kwargs: Description

    Returns:
       best_results: dictionary of resutls.
    """
    budget = 1000
    best_results = super().compute(config, budget, *args, **kwargs)

    return best_results

  @staticmethod
  def get_configspace():
    """Get configuration for a experiment run.

    This staticmethod provides the configuration space to perform sampling from.

    Returns:
        cs: Configspace with all hyperparameters.
    """
    cs = CS.ConfigurationSpace()

    samples_per_class = CSH.CategoricalHyperparameter('samples_per_class',
                                                      [2, 5, 10, 20])
    cs.add_hyperparameters([samples_per_class])
    all_models_names = ['LabelPropSparseGATMultilabelFinal']
    model = CSH.CategoricalHyperparameter('model', all_models_names)
    datasets = ['ppi']
    num_layers = CSH.CategoricalHyperparameter('num_layers', [2])
    data = CSH.CategoricalHyperparameter('data', datasets)

    lr = CSH.CategoricalHyperparameter('lr', [.005])

    patience = CSH.CategoricalHyperparameter('patience', [100])

    repeated_runs = CSH.CategoricalHyperparameter('repeated_runs', range(5))
    cs.add_hyperparameters(
        [model, data, num_layers, patience, lr, repeated_runs])

    if 'graphsage' in all_models_names:
      cs = add_graphsage_config(cs, model)

    pseudo_label_mode = CSH.CategoricalHyperparameter('pseudo_label_mode',
                                                      ['final_only'])
    label_prop_steps = CSH.CategoricalHyperparameter('label_prop_steps',
                                                     range(1, 5))
    cs.add_hyperparameters([pseudo_label_mode, label_prop_steps])

    pooling_residual = CSH.CategoricalHyperparameter('pooling_residual', [True])
    cs.add_hyperparameters([pooling_residual])
    return cs


if __name__ == '__main__':
  """Example usage."""

  seeds_array = [5006, 690, 42]
  worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)
  worker.repeat_per_run = 1
  cs = worker.get_configspace()
  config = cs.sample_configuration().get_dictionary()
  config['samples_per_class'] = 1
  config['data'] = 'ppi'
  config['model'] = 'LabelPropGATMultilabelFinal'
  config['use_lp_logits'] = True
  config['temp'] = 1

  config['label_prop_steps'] = 1
  config['pooling_residual'] = True
  config['transform'] = 'none'
  config['ignore_last_att_weights'] = False

  BUDGET = 1

  print(config)
  res = worker.compute(config=config, budget=BUDGET, working_directory='.')
  print(res)
