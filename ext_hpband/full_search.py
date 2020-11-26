"""Implementation of full-search."""
import time
import copy

import numpy as np

from hpbandster.core.master import Master
from ext_hpband.full_sampling import FullSampling as RS
from ext_hpband.full_iteration import FullIteration
from hpbandster.core.result import Result


class FullSearch(Master):
  """FullSearch for automl.

  Attributes
  ----------
  budget_per_iteration : int
      How many epochs per iterations.
  budgets : int
      Total budgets.
  eta : int
      Spliting paramater for hypeband search.
  max_budget : int
      Maximum amount of epochs for optimization.
  max_SH_iter : float
      internal paramater of BOHB-optimizer
  min_budget : int
      How many epochs to train at least.
  time_ref : time
      internal paramater to track time.
  """

  def __init__(self,
               configspace=None,
               eta=3,
               min_budget=1,
               max_budget=1,
               **kwargs):
    """Implements a random search across the search space for

    comparison.

    Candidates are sampled at random and run on the maximum budget.

    Args
    ----------
      configspace : ConfigSpace object
        valid representation of the search space
      eta : float
        In each iteration, a complete run of sequential halving is executed. In
        it,
        after evaluating each configuration on the same subset size, only a
        fraction of
        1/eta of them 'advances' to the next round.
        Must be greater or equal to 2.
      min_budget : int, optional
          Min epochs for training
      max_budget : int, optional
          Max epochs for training
      **kwargs
          Additional paramaters.

    Raises
    ------
      ValueError
          The configspace needs to provides valid objects.
    """

    if configspace is None:
      raise ValueError('You have to provide a valid ConfigSpace object')

    cg = RS(configspace=configspace, previous_result=kwargs['previous_result'])

    super().__init__(config_generator=cg, **kwargs)

    # Hyperband related stuff
    self.eta = eta
    self.min_budget = max_budget
    self.max_budget = max_budget

    # precompute some HB stuff
    self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    self.budgets = max_budget * np.power(
        eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))

    # max total budget for one iteration
    self.budget_per_iteration = 1000 * 1e5

    self.config.update({
        'eta': eta,
        'min_budget': max_budget,
        'max_budget': max_budget,
    })

  def get_next_iteration(self, iteration, iteration_kwargs={}):
    """Returns a SH iteration with only evaluations on the biggest budget

    Args
    ----------
      iteration : int
        the index of the iteration to be instantiated
      iteration_kwargs : dict, optional
          Description

    Returns
    -------
      SuccessiveHalving : the SuccessiveHalving iteration with the
        corresponding number of configurations
    """

    budgets = [self.max_budget]
    ns = [self.budget_per_iteration // self.max_budget]

    return (FullIteration(
        HPB_iter=iteration,
        num_configs=ns,
        budgets=budgets,
        config_sampler=self.config_generator.get_config,
        **iteration_kwargs))

  def run(
      self,
      n_iterations=1,
      min_n_workers=1,
      iteration_kwargs={},
  ):
    """Run n_iterations of SuccessiveHalving.


    Args: ----------
      n_iterations : int number of iterations to be performed in this run.
      min_n_workers : int minimum number of workers before starting the run.
      iteration_kwargs : dict, optional Some keyword-arguments to configure the
      run.
    """

    self.wait_for_workers(min_n_workers)

    iteration_kwargs.update({'result_logger': self.result_logger})

    if self.time_ref is None:
      self.time_ref = time.time()
      self.config['time_ref'] = self.time_ref

      self.logger.info('HBMASTER: starting run at %s' % (str(self.time_ref)))

    self.thread_cond.acquire()
    while True:

      self._queue_wait()

      next_run = None
      # find a new run to schedule
      for i in self.active_iterations():
        next_run = self.iterations[i].get_next_run()
        if not next_run is None:
          break
      if next_run == -1:
        #in case of full-evaluation, we need to stop right there
        break
      if not next_run is None:
        self.logger.debug('HBMASTER: schedule new run for iteration %i' % i)
        self._submit_job(*next_run)
        continue
      else:
        if n_iterations > 0:  #we might be able to start the next iteration
          self.iterations.append(
              self.get_next_iteration(len(self.iterations), iteration_kwargs))
          n_iterations -= 1
          continue

      # at this point there is no imediate run that can be scheduled,
      # so wait for some job to finish if there are active iterations
      if self.active_iterations():
        self.thread_cond.wait()
      else:
        break

    self.thread_cond.release()

    for i in self.warmstart_iteration:
      i.fix_timestamps(self.time_ref)

    ws_data = [i.data for i in self.warmstart_iteration]

    return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data,
                  self.config)
