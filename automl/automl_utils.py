"""This module provides some utilizations for config of automl."""
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS


def add_sparsemax_config(cs):
  """Add new configs for sparsemax model.

  This fucntions adds hyperparameters specifically for sparsemax.

  Args:
      cs (Configspace): automl-object to add auto-ml-parameters.

  Returns:
      cs (Configspace): reference to the configspace with added params.
  """
  lambda_sparsemax = CSH.CategoricalHyperparameter(
      'lambda_sparsemax', [1e-10, 1e-5, .1, 1, 2.5, 5, 10, 20], default_value=1)

  attn_drop = CSH.CategoricalHyperparameter('attn_drop', [0, .3, .6])
  in_drop = CSH.CategoricalHyperparameter('in_drop', [0, .3, .6])
  att_last_layer = CSH.CategoricalHyperparameter('att_last_layer',
                                                 [True, False])

  hyperparameters = [in_drop, att_last_layer, attn_drop, lambda_sparsemax]
  cs.add_hyperparameters(hyperparameters)

  return cs


def add_gat_config(cs, model):
  """Add new cnofigs for gat models.

  This fucntions adds hyperparameters specifically for GAT.

  Args:
      cs (Configspace): automl-object to add auto-ml-parameters.
      model (str): which model to be used.

  Returns:
      cs (Configspace): reference to the configspace with added params.
  """
  attn_drop = CSH.CategoricalHyperparameter('attn_drop', [0, .3, .6])
  in_drop = CSH.CategoricalHyperparameter('in_drop', [0, .3, .6])
  hyperparameters = [in_drop, attn_drop]
  cs.add_hyperparameters(hyperparameters)
  for var in hyperparameters:
    cond = CS.EqualsCondition(var, model, 'gat')
    cs.add_condition(cond)

  return cs


def add_dropout_ignore_gat(cs, model_hyperparam, model_names):
  """Add dropout for all models except for GAT-variants.

  The name of dropout hyperparameters is different for GAT-variants. Therefore,
  we need a special treatment for these models.

  Args:
      cs (Configspace): automl-object to add auto-ml-parameters.
      model_hyperparam (hyperparameters): hyperparameters-object for automl
      model_names (str): which model to be used

  Returns:
      cs (Configspace): reference to the configspace with added params.
  """
  dropout = CSH.CategoricalHyperparameter('dropout', [0, .3, .6])
  cs.add_hyperparameters([dropout])
  ignored_models = ['gat', 'sparselossgat', 'UnsupSparseGAT']
  models_with_dropout = list((set(model_names) - set(ignored_models)))
  cond_list = []
  for model_name in models_with_dropout:
    cond = CS.EqualsCondition(dropout, model_hyperparam, model_name)
    cond_list.append(cond)
  if len(cond_list) > 1:
    cs.add_condition(CS.OrConjunction(*cond_list))
  else:
    cs.add_condition(*cond_list)
  return cs


def add_cond_model_name(cs, hyperparameters, model, model_name):
  """Provides a generic customizer for a new hyperparameter.

  Based on the model-hyperparameter and model name, we configure the configspace
  so that it is only activated for certain models.

  Args:
      cs (Configspace): automl-object to add auto-ml-parameters.
      hyperparameters (hyperparameters): auto-ml-parameters.
      model (hyperparameters): auto-ml-parameters for model.
      model_name (str): which model for usage.

  Returns:
      cs (Configspace): reference to the configspace with added params.
  """
  for var in hyperparameters:
    cond_list = []
    for model_name in [model_name]:
      cond = CS.EqualsCondition(var, model, model_name)
      cond_list.append(cond)
    if len(cond_list) > 1:
      cs.add_condition(CS.OrConjunction(*cond_list))
    else:
      cs.add_condition(*cond_list)
  return cs


def add_graphsage_config(cs, model):
  """Adds some configs needed for graphsage.

  This fucntion adds config for the graphsage

  Args:
      cs (Configspace): automl-object to add auto-ml-parameters.
      model (hyperparameters): the auto-ml-parameters for model.

  Returns:
      cs (Configspace): reference to the configspace with added params.
  """
  aggregator_type = CSH.CategoricalHyperparameter('graphsage_aggregator_type',
                                                  ['gcn', 'mean', 'pool'])
  hyperparameters = [aggregator_type]
  cs.add_hyperparameters(hyperparameters)

  cs = add_cond_model_name(cs, hyperparameters, model, 'graphsage')

  return cs
