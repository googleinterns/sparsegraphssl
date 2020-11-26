"""Tests labelprop-gats."""
import pytest
from run_scripts.gat_train import main as main_ctgr
from run_scripts.gat_train import get_args as get_args_ctgr


@pytest.fixture
def base_test_config_ctgr():
  """Summary

  Returns:
      TYPE: Description
  """
  args = get_args_ctgr().parse_args([])
  config = args.__dict__
  config['samples_per_class'] = 1
  config['label_prop_steps'] = 3
  config['in_drop'] = 0
  config['attn_drop'] = 0
  config['data'] = 'cora'
  config['repeated_runs'] = 0
  config['model'] = 'LabelPropSparseGAT'
  config['patience'] = 20
  config['pooling_residual'] = True
  config['use_adj_matrix'] = False
  config['gpu'] = -1
  return args


@pytest.mark.labelprop
def test_gat_lp(base_test_config_ctgr):
  """Tests GAT-LP.

  Args:
      base_test_config_ctgr (fixture): a standard test config.
  """
  args = base_test_config_ctgr
  args.model = 'LabelPropGAT'
  res = main_ctgr(args)
  test_acc = res[0]
  assert test_acc > .38


@pytest.mark.labelprop
def test_sign_lp(base_test_config_ctgr):
  """Tests SIGN-LP

  Args:
      base_test_config_ctgr (fixture): a standard test config.
  """
  args = base_test_config_ctgr
  args.model = 'LabelPropSparseGAT'
  res = main_ctgr(args)
  test_acc = res[0]
  assert test_acc > .38


@pytest.mark.baselines
def test_gat(base_test_config_ctgr):
  """Test GAT.

  Args:
      base_test_config_ctgr (fixture): a standard test config.
  """
  args = base_test_config_ctgr
  args.model = 'GAT'
  res = main_ctgr(args)
  test_acc = res[0]
  assert test_acc > .20


@pytest.mark.baselines
def test_sign(base_test_config_ctgr):
  """Tests SIGN.

  Args:
      base_test_config_ctgr (fixture): a standard test config.
  """
  args = base_test_config_ctgr
  args.model = 'SparseGAT'
  res = main_ctgr(args)
  test_acc = res[0]
  assert test_acc > .20
