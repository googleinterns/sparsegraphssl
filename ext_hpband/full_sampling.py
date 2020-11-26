"""Sampling process of automl."""

from hpbandster.core.base_config_generator import base_config_generator
import time


class FullSampling(base_config_generator):
  """Class to implement random sampling from a ConfigSpace.

  Attributes:
      configspace (Configspace): all configuration space.
      previous_result (Results): container with all results.
      used_space (set): set of hyperparameter space which has been used.
  """

  def __init__(self, configspace, previous_result, **kwargs):
    """Args:

        configspace: ConfigSpace.ConfigurationSpace The configuration space to
        sample from. It contains the full specification of the Hyperparameters
        with their priors
        previous_result (TYPE): Description
        **kwargs: see  hyperband.config_generators.base.base_config_generator
        for additional arguments -----------
    """

    super().__init__(**kwargs)
    self.configspace = configspace
    self.used_space = set()
    self.previous_result = previous_result

  def get_config(self, budget):
    """Retrieves the configuration.

    Args:
        budget (int): how many epochs for training.

    Returns:
        new_config: new configuration to run next.
    """
    if self.previous_result is not None:
      id2conf = self.previous_result.get_id2config_mapping()
      finished_runs = self.previous_result.get_all_runs()
      finished_configs = [
          id2conf[c['config_id']]['config']
          for c in finished_runs
          if c['config_id'] != {}
      ]

      for config in finished_configs:
        sorted_keys = list(config.keys())
        sorted_keys.sort()
        values = [config[key] for key in sorted_keys]
        config_values = tuple(values)
        self.used_space.add(config_values)

    start_time = time.time()
    config_found = False
    while not config_found:
      new_config = self.configspace.sample_configuration().get_dictionary()
      sorted_keys = list(new_config.keys())
      sorted_keys.sort()
      values = [new_config[key] for key in sorted_keys]
      config_values = tuple(values)
      if config_values not in self.used_space:
        self.used_space.add(config_values)
        config_found = True
      time.sleep(.01)

      total_time_elapsed = time.time() - start_time
      if total_time_elapsed > 10:
        return (-1, {})  # we are at the end.

    #warnings.warn('ends after 5 seconds heer ')
    return (new_config, {})
