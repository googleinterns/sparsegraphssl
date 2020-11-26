"""Summary"""
import pytest
import dgl
from dgl.nn.pytorch.softmax import edge_softmax
import torch as th
from model_archs.operations.edge_sparsemax import edge_sparsemax
from data_reader.data_utils import read_pickle
import numpy as np


def test_edgesoftmax():
  """Tests edge softmax function."""

  g = dgl.DGLGraph()
  g.add_nodes(3)
  g.add_edges([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
  edata = th.ones(6, 1).float()
  expected = th.Tensor([[1.], [1.], [1.], [1.], [1.], [1.]])
  assert th.isclose(edata, expected, atol=1e-2).sum() == len(expected)

  # Apply edge softmax on g:

  results = edge_softmax(g, edata)
  expected = th.Tensor([[1.0000], [0.5000], [0.3333], [0.5000], [0.3333],
                        [0.3333]])
  assert th.isclose(results, expected, atol=1e-2).sum() == len(expected)
  # Apply edge softmax on first 4 edges of g:

  results = edge_softmax(g, edata[:4], th.Tensor([0, 1, 2, 3]))
  expected = th.Tensor([[1.0000], [0.5000], [1.0000], [0.5000]])
  assert th.isclose(results, expected, atol=1e-2).sum() == len(expected)


def _make_ix_like(input, dim=0):
  """Constructs index vector.

  Args:
      input (tensor): input features.
      dim (int, optional): which dim to perform operations over.

  Returns:
      index_vector: reshaped to match required dimensions..
  """
  d = input.size(dim)
  rho = th.arange(1, d + 1, device=input.device, dtype=input.dtype)
  view = [1] * input.dim()
  view[0] = -1
  return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=1):
  """Sparsemax building block: compute the threshold

  Use this directly on input = [a0, a1, a2...]

  Args:
      input: any dimension
      dim: dimension along which to apply the sparsemax

  Returns:
      the threshold value
  """
  input /= 5
  input_srt, _ = th.sort(input, descending=True, dim=dim)
  input_cumsum = input_srt.cumsum(dim) - 1
  rhos = _make_ix_like(input, dim)
  support = rhos * input_srt > input_cumsum

  support_size = support.sum(dim=dim).unsqueeze(dim)
  tau = input_cumsum.gather(dim, support_size - 1)
  tau /= support_size.to(input.dtype)
  return tau, support_size


def forward(input):
  """Sparsemax: normalizing sparse transform (a la softmax)

  Parameters:
      input (Tensor): any shape

  Returns:
      output (Tensor): same shape as input

  Deleted Parameters:
      dim: dimension along which to apply sparsemax
  """
  # ctx.dim = dim
  dim = 0
  max_val, _ = input.max(dim=dim, keepdim=True)
  input -= max_val  # same numerical stability trick as for softmax
  tau, supp_size = _threshold_and_support(input, dim=dim)
  output = th.clamp(input - tau, min=0)
  # ctx.save_for_backward(supp_size, output)
  return output


@pytest.mark.sparsemax
def test_sparsemax():
  """Tests sparsemax implementation."""

  g = dgl.DGLGraph()
  g.add_nodes(3)
  g.add_edges([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
  # edata = th.ones(6, 1).float()
  edata = th.Tensor([[1], [1], [0], [1], [1.1], [0]]).float()

  single_edata = th.Tensor([[0], [1.1], [0]]).float()
  print(forward(single_edata))

  results = edge_sparsemax(g, edata)
  print(results)


def backward(grad_score, supp_size, output):
  """Tests backward function

  Args:
      grad_score (tensor): grad_score wrt. outputs.
      supp_size (tensor): computed support size of sparsemax.
      output (tensor): output during forward pass.

  Returns:
      grad_input: gradients wrt. intputs
  """
  dim = 1

  msg_sum = th.sum(grad_score, dim=1)
  supp_size = supp_size.view(msg_sum.shape)
  v_hat = msg_sum / supp_size

  v_hat = v_hat.unsqueeze(dim)
  grad_input = th.where(output != 0, grad_score - v_hat, grad_score)
  return grad_input


@pytest.mark.check_grad
def test_gradient():
  """Test gradient of sparsemax."""
  grad_score, out, supp_size, v_hat, nodes_ids = read_pickle('all.pkl')
  grad_input_minus_vhat = grad_score - v_hat
  grad_input = th.where(out != 0, grad_input_minus_vhat, grad_score)

  correct_grad = backward(grad_score, supp_size, out)
  assert th.isclose(
      grad_input, correct_grad, atol=1e-2).sum() == np.prod(grad_score.shape)
