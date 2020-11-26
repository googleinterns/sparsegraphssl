"""Some utility functions for the sparsemax operations."""
# Lint as python3

import torch


def reduce_tau_support_size(nodes):
  """Computes threshold tau and support size.

  This function performs thresholding over incoming signals on the nodes.

  Args:
    nodes: container of all information on nodes.

  Returns:
    results: dict with saved results.
  """
  tau, support_size = _threshold_and_support(nodes.mailbox['norm_score'], dim=1)
  results = {'tau': tau, 'support_size': support_size}
  return results


def edge_clamp(edges):
  """Clamps the signals on edges.

   This function sets the values which are smaller than 0 to be 0. Applied to
   all edges.

  Args:
    edges: container of all information on edges.

  Returns:
    results: dict with saved results.
  """
  tau = edges.dst['tau']
  input_feat = edges.data['norm_score']
  output = torch.clamp(input_feat - tau.view(input_feat.shape), min=0)
  results = {'out': output}
  return results


def _make_ix_like(input_feat, dim=0):
  """Creates an index array from 1 to k.

  This function computes an index array from 1 to k to compute the weighted
  cum-sum
  operation in SparseMax.


  Args:
    input_feat: input features
    dim: dimension to perform the thresholding operation over.

  Returns:
    index array: similar shape to input_feat
  """
  d = input_feat.size(dim)
  rho = torch.arange(1, d + 1, device=input_feat.device, dtype=input_feat.dtype)
  view = [1] * input_feat.dim()
  view[0] = -1
  return rho.view(view).transpose(0, dim)


def _threshold_and_support(input_feat, dim=1):
  """Sparsemax building block: compute the threshold.

  This function computes the threshold and support for the incoming input
  features.

  Args:
    input_feat: input features
    dim: dimension to perform the thresholding operation over.

  Returns:
    the threshold value
  """

  input_feat_srt, _ = torch.sort(input_feat, descending=True, dim=dim)
  input_feat_cumsum = input_feat_srt.cumsum(dim) - 1
  rhos = _make_ix_like(input_feat, dim)
  support = rhos * input_feat_srt > input_feat_cumsum
  support_size = support.sum(dim=dim).unsqueeze(dim)
  assert (support_size == 0).sum() == 0, 'prevents mem-break'
  tau = input_feat_cumsum.gather(dim, support_size - 1)
  tau /= support_size.to(input_feat.dtype)
  return tau, support_size
