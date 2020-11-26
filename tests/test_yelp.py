import pytest
from automl.worker_yelp import PyTorchWorker


@pytest.mark.simple
def test_simple():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegat'
    # config['attention_loss'] = 'sparse_max'
    config['main_loss'] = 'entropy_only'
    config['data'] = 'yelp_rur'
    # config['samples_per_class'] = 16
    config['samples_per_class'] = 10
    config['attn_drop'] = 0
    config['in_drop'] = 0
    config['repeated_runs'] = 0
    config['lr'] = 0.01
    config['patience'] = 200
    config['num_heads'] = 8
    config['num_out_heads'] = 1
    config['num_hidden'] = 50
    config['batch_size'] = 1
    # config['lambda_sparsemax'] = 0

    # config['use_exact_n_hops'] = True
    # config['n_hops_max'] = 2
    budget = 10
    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)
@pytest.mark.gat
def test_gat():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=-2, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    # config['model'] = 'sparsegat'
    config['model'] = 'gat'
    config['attention_loss'] = 'sparse_max'
    config['attention_weight'] = 10
    config['main_loss'] = 'entropy_only'
    config['data'] = 'yelp_rur'
    config['samples_per_class'] = 100
    config['attn_drop'] = 0
    config['in_drop'] = 0
    config['repeated_runs'] = 0
    config['lr'] = 0.001
    config['patience'] = 1000
    config['num_heads'] = 8
    config['num_layers'] = 2
    config['residual'] = True
    config['num_out_heads'] = 1
    config['num_hidden'] = 50
    config['batch_size'] = 1
    # config['lambda_sparsemax'] = 0

    # config['use_exact_n_hops'] = True
    # config['n_hops_max'] = 2
    budget = 1000
    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)