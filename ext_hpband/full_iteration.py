"""Exention of automl to support full-grid search."""
from hpbandster.core.base_iteration import BaseIteration
from hpbandster.core.base_iteration import Datum

import time


class FullIteration(BaseIteration):
  """Full-grid search with automl."""

  def get_next_run(self):
    """Returns the next configuration and budget to run.

    This function is called from HB_master, don't call this from
    your script.

    It returns None if this run of SH is finished or there are
    pending jobs that need to finish to progress to the next stage.

    If there are empty slots to be filled in the current SH stage
    (which never happens in the original SH version), a new
    configuration will be sampled and scheduled to run next.

    Returns
    -------
    next_run
        Next configuration to run.
    """

    if self.is_finished:
      return (None)

    for k, v in self.data.items():
      if v.status == 'QUEUED':
        assert v.budget == self.budgets[
            self
            .stage], 'Configuration budget does not align with current stage!'
        v.status = 'RUNNING'
        self.num_running += 1
        return (k, v.config, v.budget)

    # check if there are still slots to fill in the current stage and return that
    if (self.actual_num_configs[self.stage] < self.num_configs[self.stage]):
      anything_left = self.add_configuration()
      print('running jobs {} '.format(self.num_running))
      if anything_left == -1:
        if self.num_running != 0:
          # the jobs have a callback to reduce self.num_running when they are finished
          time.sleep(300)
          return (self.get_next_run())
        self.process_results()
        return -1
      else:
        return (self.get_next_run())
    if self.num_running == 0:
      # at this point a stage is completed
      self.process_results()
      return (self.get_next_run())

    return (None)

  def add_configuration(self, config=None, config_info={}):
    """Adds a new configuration to the current iteration.

    Args
    ----------
      config : valid configuration
        The configuration to add. If None, a configuration is sampled from the
        config_sampler
      config_info : dict
        Some information about the configuration that will be stored in the
        results

    Returns
    -------
    config_id
        id of configuration.

    Raises
    ------
    RuntimeError
        If no other configuration can be added.
    """

    if config is None:
      config, config_info = self.config_sampler(self.budgets[self.stage])
      if config == -1:
        return -1

    if self.is_finished:
      raise RuntimeError(
          "This HPBandSter iteration is finished, you can't add more configurations!"
      )

    if self.actual_num_configs[self.stage] == self.num_configs[self.stage]:
      raise RuntimeError(
          "Can't add another configuration to stage %i in HPBandSter iteration %i."
          % (self.stage, self.HPB_iter))

    config_id = (self.HPB_iter, self.stage, self.actual_num_configs[self.stage])

    self.data[config_id] = Datum(
        config=config, config_info=config_info, budget=self.budgets[self.stage])

    self.actual_num_configs[self.stage] += 1

    if not self.result_logger is None:
      self.result_logger.new_config(config_id, config, config_info)

    return (config_id)
