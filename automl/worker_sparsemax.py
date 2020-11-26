# from hpbandster.core.worker import Worker

from automl.worker_nclass import PyTorchWorker as NClassWorker
from automl.worker_nclass import add_sparsegat_config
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch
from automl.datasets_settings import elliptic_settings
from automl.datasets_settings import citation_graphs_settings
from automl.datasets_settings import yelp_settings
from automl.datasets_settings import protein_settings
from dgl_extensions.gat_train_ppi import main as main_ppi
from dgl_extensions.gat_train import main as main_ctgr
import gc

class PyTorchWorker(NClassWorker):

    def __init__(self, **kwargs):
        # pdb.set_trace()

        super().__init__(**kwargs)
        gc.collect() 
        torch.cuda.empty_cache() 

    def compute(self, config, budget, *args, **kwargs):

        if config['data'] in ['cora', 'citeseer', 'pubmed']:
            config.update(citation_graphs_settings)
            self.config = self.add_config(config, budget)
            best_test_perf, results = main_ctgr(self.config)
        elif config['data'] in ['ppi', 'elliptic']:
            config.update(citation_graphs_settings)
            self.config = self.add_config(config, budget)
            best_test_perf, results = main_ppi(self.config)
        elif config['data'].startswith('yelp'):
            config.update(yelp_settings)
        elif config['data'] == 'protein':
            config.update(protein_settings)
            config['num_epochs'] = 2000
            results = self.run_citation_graph(config, budget)
        else:
            raise NotImplementedError
        gc.collect() 
        torch.cuda.empty_cache() 

        best_epoch,  best_test_perf, _, best_valid_perf, _ = results
        best_results = ({
              'loss': 1-best_valid_perf, # remember: HpBandSter always minimizes!
              'info': { 'best_epoch': best_epoch,
                        'best_test_perf': best_test_perf,
                        'best_test_perf_f1': best_test_perf,
                        'best_valid_perf': best_valid_perf,
                        'best_valid_perf_f1': best_valid_perf,
                        }
                        })
        return best_results
    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        # easy to start with

        # model = CSH.CategoricalHyperparameter('model', ['sparsegat', 'gat'])
        # datasets = ['cora', 'citeseer', 'pubmed']

        model = CSH.CategoricalHyperparameter('model', ['sparsegat'])
        datasets = ['cora', 'citeseer']

        # datasets = ['elliptic']
        data = CSH.CategoricalHyperparameter('data', datasets)

        num_layers = CSH.CategoricalHyperparameter('num_layers', [2])
        repeated_runs = CSH.CategoricalHyperparameter('repeated_runs', range(10))
        lr = CSH.CategoricalHyperparameter('lr', [.001, 0.005])

        # n_hops_max = CSH.CategoricalHyperparameter('n_hops_max', [2, 3])
        # use_exact_n_hops = CSH.CategoricalHyperparameter('use_exact_n_hops', [True, False])
        labeling_rate = CSH.CategoricalHyperparameter('labeling_rate', [0.05])
        patience = CSH.CategoricalHyperparameter('patience', [40])

        lambda_sparsemax = CSH.CategoricalHyperparameter('lambda_sparsemax', [-100, -200, -400, -800, -1600, -3200])

        cs.add_hyperparameters([model, data,  num_layers, repeated_runs, labeling_rate, patience])
        cs.add_hyperparameters([lambda_sparsemax, lr])
        cs = add_sparsegat_config(cs, model)
        return cs


def add_sparsegat_config(cs, model):
    # attention_weight = CSH.CategoricalHyperparameter('attention_weight',
    # [1e-2, 1e-1, 1, 10, 1e2])
    attn_drop = CSH.CategoricalHyperparameter('attn_drop',
                                              [0, .3, .6])
    in_drop = CSH.CategoricalHyperparameter('in_drop',
                                            [0, .3, .6])
    attention_loss = CSH.CategoricalHyperparameter('attention_loss',
                                                   ['sparse_max'])
    main_loss = CSH.CategoricalHyperparameter('main_loss', ['entropy_and_sparsity'])
    # att_last_layer = CSH.CategoricalHyperparameter('att_last_layer', [True, False])

    hyperparameters = [in_drop, attention_loss]
    cs.add_hyperparameters(hyperparameters)
    cs.add_hyperparameters([main_loss, attn_drop])
    # for var in hyperparameters:
    #     cond_list = []
    #     for loss in ['entropy_and_sparsity']:
    #         cond = CS.EqualsCondition(var, main_loss, loss)
    #         cond_list.append(cond)
    #     if len(cond_list) > 1:
    #         cs.add_condition(CS.OrConjunction(*cond_list))
    #     else:
    #         cs.add_condition(*cond_list)

    return cs


if __name__ == "__main__":
    # seeds_array = [5006, 690, 9878, 9073, 7868, 1316]

    seeds_array = [5006, 690, 42]
    worker = PyTorchWorker(run_id='0', gpu=-1, seeds_array=seeds_array)

    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    budget = 1
    # '''
    config['data'] = 'pubmed'
    config['model'] = 'sparsegat'
    config['repeated_runs'] = 0
    config['samples_per_class'] = 1

    res = worker.compute(config=config, budget=budget, working_directory='.')
    print(res)
