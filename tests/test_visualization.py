"""Summary

Attributes:
    data (list): Description
    gpu (list): Description
    model (list): Description
    n_nodes_to_plot (list): Description
    samples_per_class (list): Description
"""
import pytest
from run_scripts.gat_train import main as main_ctgr
from run_scripts.gat_train import get_args as get_args_ctgr
import itertools


@pytest.fixture
def base_test_config_ctgr():
  """Base configuration for testing

  Returns:
      fixture: Standard configuration.
  """
  args = get_args_ctgr().parse_args([])
  config = args.__dict__
  config['samples_per_class'] = 1
  config['label_prop_steps'] = 3
  config['in_drop'] = 0
  config['attn_drop'] = 0

  config['data'] = 'citeseer'
  config['repeated_runs'] = 0
  config['model'] = 'LabelPropSparseGAT'
  config['pooling_residual'] = True
  config['use_adj_matrix'] = False
  config['gpu'] = -1
  # args.vis_att = True
  args.vis_att = False
  args.att_analysis = False

  args.patience = 50
  args.epochs = 100
  return args


samples_per_class = [2]
data = ['cora']

n_nodes_to_plot = [None]
gpu = [-1]
model = ['SparseGAT']


@pytest.mark.tree
@pytest.mark.parametrize('samples_per_class, data, gpu,  model',
                         itertools.product(samples_per_class, data, gpu, model))
def test_tree(base_test_config_ctgr, samples_per_class, data, gpu, model):
  """Tests tree for interpretability.

  Args:
      base_test_config_ctgr (fixture): Standard fixture for parameters settings.
      samples_per_class (int): how many samples/class to be used.
      data (str): which dataset to use.
      gpu (int): which gpu to use.
      n_nodes_to_plot (int): how many nodes to plot
      model (TYPE): Description
  """
  args = base_test_config_ctgr
  args.samples_per_class = samples_per_class
  args.data = data
  args.gpu = gpu
  args.model = 'GAT'
  args.lambda_sparsemax = 0  # .9999
  args.label_prop_steps = 2
  args.epochs = 100
  args.compute_tree_sparsification = False
  args.plot_tree = True
  # args.use_adj_matrix = True
  res = main_ctgr(args)


@pytest.mark.shortest_path
@pytest.mark.parametrize('samples_per_class, data, gpu, n_nodes_to_plot',
                         itertools.product(samples_per_class, data, gpu,
                                           n_nodes_to_plot))
def test_analyze_shortest_paths(base_test_config_ctgr, samples_per_class, data,
                                gpu, n_nodes_to_plot):
  """Tests attention on shortest_paths

  Args:
      base_test_config_ctgr (fixture): Standard fixture for parameters settings.
      samples_per_class (int): how many samples/class to be used.
      data (str): which dataset to use.
      gpu (int): which gpu to use.
      n_nodes_to_plot (int): how many nodes to plot
  """
  args = base_test_config_ctgr
  args.samples_per_class = samples_per_class
  args.n_nodes_to_plot = n_nodes_to_plot
  args.data = 'citeseer'
  args.gpu = gpu
  args.model = 'GAT'
  args.lambda_sparsemax = 0
  args.epochs = 1

  # args.num_heads = 1
  args.label_prop_steps = 3
  args.noise_robustness = False
  args.att_analysis = False
  args.noise_injected_ratio = 0
  args.noise_type = 'heterophily'  #.2
  args.att_on_shortest_paths = True
  # args.epochs = 100
  res = main_ctgr(args)
  test_acc = res[0]


@pytest.mark.noise_robustness
@pytest.mark.parametrize('samples_per_class, data, gpu, n_nodes_to_plot',
                         itertools.product(samples_per_class, data, gpu,
                                           n_nodes_to_plot))
def test_analyze_gat(base_test_config_ctgr, samples_per_class, data, gpu,
                     n_nodes_to_plot):
  """Test noise robustness on GAT.

  Args:
      base_test_config_ctgr (fixture): Standard fixture for parameters settings.
      samples_per_class (int): how many samples/class to be used.
      data (str): which dataset to use.
      gpu (int): which gpu to use.
      n_nodes_to_plot (int): how many nodes to plot
  """
  args = base_test_config_ctgr
  args.samples_per_class = samples_per_class
  args.n_nodes_to_plot = n_nodes_to_plot
  args.data = data
  args.gpu = gpu
  args.model = 'LabelPropGAT'
  args.lambda_sparsemax = 0.999999999999  #.9#.9 # .50
  args.epochs = 100

  args.label_prop_steps = 1
  args.noise_robustness = True
  args.att_analysis = False
  args.noise_injected_ratio = 0
  res = main_ctgr(args)
  test_acc = res[0]


@pytest.mark.important
@pytest.mark.gat
@pytest.mark.parametrize('samples_per_class, data, gpu, n_nodes_to_plot',
                         itertools.product(samples_per_class, data, gpu,
                                           n_nodes_to_plot))
def test_visualize_gat(base_test_config_ctgr, samples_per_class, data, gpu,
                       n_nodes_to_plot):
  """Tests visualization of GAT.

  Args:
      base_test_config_ctgr (fixture): Standard fixture for parameters settings.
      samples_per_class (int): how many samples/class to be used.
      data (str): which dataset to use.
      gpu (int): which gpu to use.
      n_nodes_to_plot (int): how many nodes to plot
  """
  args = base_test_config_ctgr
  args.samples_per_class = samples_per_class
  args.n_nodes_to_plot = n_nodes_to_plot
  args.data = data
  args.gpu = gpu
  args.model = 'GAT'

  res = main_ctgr(args)
  test_acc = res[0]


@pytest.mark.important
@pytest.mark.sign
@pytest.mark.parametrize('samples_per_class, data, gpu, n_nodes_to_plot',
                         itertools.product(samples_per_class, data, gpu,
                                           n_nodes_to_plot))
def test_visualize_sign(base_test_config_ctgr, samples_per_class, data, gpu,
                        n_nodes_to_plot):
  """Tests visualization of SIGN.

  Args:
      base_test_config_ctgr (fixture): Standard fixture for parameters settings.
      samples_per_class (int): how many samples/class to be used.
      data (str): which dataset to use.
      gpu (int): which gpu to use.
      n_nodes_to_plot (int): how many nodes to plot
  """
  args = base_test_config_ctgr
  args.samples_per_class = samples_per_class
  args.n_nodes_to_plot = n_nodes_to_plot
  args.data = data
  args.gpu = gpu
  args.model = 'SparseGAT'

  res = main_ctgr(args)
  test_acc = res[0]
  # assert test_acc > .20


@pytest.mark.lp_gat
@pytest.mark.parametrize('samples_per_class, data, gpu, n_nodes_to_plot',
                         itertools.product(samples_per_class, data, gpu,
                                           n_nodes_to_plot))
def test_visualize_gat_lp(base_test_config_ctgr, samples_per_class, data, gpu,
                          n_nodes_to_plot):
  """Tests visualization of LP-GAT

  Args:
      base_test_config_ctgr (fixture): Standard fixture for parameters settings.
      samples_per_class (int): how many samples/class to be used.
      data (str): which dataset to use.
      gpu (int): which gpu to use.
      n_nodes_to_plot (int): how many nodes to plot
  """
  args = base_test_config_ctgr
  args.model = 'LabelPropGAT'
  args.data = data
  args.gpu = 1
  args.samples_per_class = samples_per_class
  args.n_nodes_to_plot = n_nodes_to_plot
  res = main_ctgr(args)
  test_acc = res[0]


@pytest.mark.lp_sign
@pytest.mark.parametrize('samples_per_class, data, gpu, n_nodes_to_plot',
                         itertools.product(samples_per_class, data, gpu,
                                           n_nodes_to_plot))
def test_visualize_sign_lp(base_test_config_ctgr, samples_per_class, data, gpu,
                           n_nodes_to_plot):
  """Tests visualization of LP-SIGN

  Args:
      base_test_config_ctgr (fixture): Standard fixture for parameters settings.
      samples_per_class (int): how many samples/class to be used.
      data (str): which dataset to use.
      gpu (int): which gpu to use.
      n_nodes_to_plot (int): how many nodes to plot
  """
  args = base_test_config_ctgr
  args.samples_per_class = samples_per_class
  args.n_nodes_to_plot = n_nodes_to_plot
  args.data = data
  args.gpu = gpu
  args.model = 'LabelPropSparseGAT'
  res = main_ctgr(args)
  test_acc = res[0]
