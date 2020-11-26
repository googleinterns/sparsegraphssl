import pytest
from automl.worker_nclass import PyTorchWorker

@pytest.fixture
def base_config():
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['data'] = 'ppi'
    config['model'] = 'sparsegat'
    budget = 1
    return (worker, config, budget)

@pytest.mark.normal_ppi
def test_L01(base_config):
    worker, config, budget = base_config
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'l0.1'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'ppi'
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.simple
def test_L02(base_config):
    worker, config, budget = base_config
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'l0.1'
    config['main_loss'] = 'entropy_and_sparsity'
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)



@pytest.mark.full_ppi
def test_L03(base_config):
    worker, config, budget = base_config
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'negl2'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'ppi'
    config['gpu'] = 2
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)
@pytest.mark.full_ppi
def test_L04(base_config):
    worker, config, budget = base_config
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'ppi'
    config['gpu'] = 3
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.full_ppi
def test_L05(base_config):
    worker, config, budget = base_config
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'l0.1'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'ppi'
    config['gpu'] = 1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.nhops_model
def test_L06(base_config):
    worker, config, budget = base_config
    config['model'] = 'sparsegatedgefeat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'ppi'
    config['gpu'] = 1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)


@pytest.mark.gpu_test
def test_L07():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'ppi'
    config['gpu'] = 1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.ctgr
def test_CTGR():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'cora'
    config['labeling_rate'] = 1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.ctgr
def test_CTGR_2():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'citeseer'
    config['labeling_rate'] = 1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)
@pytest.mark.ctgr
def test_CTGR_3():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'pubmed'
    config['labeling_rate'] = 1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.ctgr
def test_CTGR_4():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'pubmed'
    config['labeling_rate'] = .1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)



@pytest.mark.failed
def test_failed():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    failed_config = {"attn_drop": 0, "data": "cora", "labeling_rate": 0.01, "main_loss": "entropy_only", "model": "sparsegat", "num_layers": 3, "repeated_runs": 0}
    config.update(failed_config)
    # config['model'] = 'sparsegat'
    # config['attention_loss'] = 'min_entropy'
    # config['main_loss'] = 'entropy_and_sparsity'
    # config['data'] = 'pubmed'
    # config['labeling_rate'] = .1
    budget = 10
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)


@pytest.mark.ctgr_nhops
def test_CTGR_nhops_1():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegatedgefeat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'cora'
    config['labeling_rate'] = .1
    config['use_exact_n_hops'] = True
    config['n_hops_max'] = 3
    budget = 10
    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.ppi_nhops
def test_ppi_nhops():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegatedgefeat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'ppi'
    config['labeling_rate'] = .1
    config['use_exact_n_hops'] = True
    config['n_hops_max'] = 2
    budget = 10
    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)


@pytest.mark.nclass_spc
def test_spc():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'gat'
    config['attention_loss'] = 'min_entropy'
    config['main_loss'] = 'entropy_and_sparsity'
    config['data'] = 'cora'
    config['samples_per_class'] = 1
    config['repeated_runs'] = 0
    # config['use_exact_n_hops'] = True
    # config['n_hops_max'] = 2
    budget = 10
    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)



@pytest.mark.ce
def test_ce():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=2, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'sparsegat'
    # config['attention_loss'] = 'sparse_max'
    config['main_loss'] = 'entropy_only'
    config['data'] = 'citeseer'
    config['samples_per_class'] = 16
    config['attn_drop'] = 0.6
    config['in_drop'] = 0.6
    config['repeated_runs'] = 0
    config['lr'] = 0.001
    config['patience'] = 100

    # config['use_exact_n_hops'] = True
    # config['n_hops_max'] = 2
    budget = 1000
    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.new_sampling
def test_new_sampling():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config['model'] = 'gat'
    # config['attention_loss'] = 'sparse_max'
    config['main_loss'] = 'entropy_only'
    config['data'] = 'citeseer'
    config['labeling_rate'] = .0051
    # config['attn_drop'] = 0.6
    # config['in_drop'] = 0.6
    config['repeated_runs'] = 0
    config['lr'] = 0.001
    # config['patience'] = 100

    # config['use_exact_n_hops'] = True
    # config['n_hops_max'] = 2
    budget = 1
    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.sparsemax
def test_sparsemax():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()

    config['model'] = 'sparsegat'
    # config['model'] = 'gat'
    config['attention_loss'] = 'sparse_max'
    config['main_loss'] = 'entropy_only'
    config['data'] = 'yelp_rur'
    config['labeling_rate'] = 1
    config['attn_drop'] = 0
    config['in_drop'] = 0
    config['repeated_runs'] = 0
    config['lr'] = 0.001
    config['patience'] = 100
    config['num_heads'] = 1
    # config['alpha_sparsemax'] = 0
    
    # config['use_exact_n_hops'] = True
    # config['n_hops_max'] = 2
    budget = 400

    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

@pytest.mark.unsup_sparsemax()
def test_unsup_sparsemax():
    # worker, config, budget = base_config
    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()

    config['model'] = 'unsupsparsegat'
    config['attention_loss'] = 'sparse_max'
    config['main_loss'] = 'entropy_only'
    config['data'] = 'citeseer'
    config['samples_per_class'] = 1
    config['attn_drop'] = 0
    config['in_drop'] = 0
    config['repeated_runs'] = 0
    config['lr'] = 0.001
    config['patience'] = 100
    config['alpha_sparsemax'] = 0
    
    # config['use_exact_n_hops'] = True
    # config['n_hops_max'] = 2
    budget = 100

    print(config)
    res = worker.compute(config=config, budget=budget, working_directory='.')
    assert res, 'something is wrong'
    print(res)

