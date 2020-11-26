"""Implementation of Sparsemax for the deep graph library (DGL).

This is an implementation of the sparsemax operation for the message passing
interface typically used for graph neural networks. This operation was propoped
by From Softmax to Sparsemax:A Sparse Model of Attention and Multi-Label
classification (https://arxiv.org/pdf/1602.02068.pdf). This operation is
typically used in combination with attention mechanims to get sparse activation
The operation is fully-differentiable but needs a custom implementation of
backward pass.

  Typical usage example:
  graph: dgl.DGLGraph
  logits: logits predictions on each edges, typically attention weights
  simple_edge_sparsemax(graph, logits)
"""

# Lint as python3

from dgl import function as fn
from dgl.base import ALL
from dgl.base import is_all
from model_archs.operations.op_sparsemax import edge_clamp
from model_archs.operations.op_sparsemax import reduce_tau_support_size
import torch as th


class EdgeSparsemax(th.autograd.Function):
  """Apply sparsemax over signals of incoming edges.

  Sparsemax was defined in this paper: https://arxiv.org/pdf/1602.02068.pdf).
  This class provides an efficient implementation of the forward and backward
  passes using the message passing interface (MP-I).
  Note that with MP-I, many tensors cannot simply be stacked together due to
  memory-efficiency.
  For example, the edges are represented as an edge list, instead of using an
  adjacency matrix. Edges features cannot be simply stacked either, since
  nodes connectivity is typically not constant for all nodes.
  Therefore, we need to implement this SparseMax operation using the functional
  interface of DGL.
  """

  @staticmethod
  def forward(ctx, graph, score, eids):
    """Applies sparsemax to incoming scores on the edges of the graph.

    Sparsemax: Sort the elements, compute the thershold tau, shift all
    values by the thershold tau, set values <0 to 0.
    Afterwards, save the grpah, the sparse_outputs, and the support size
    in the ctx.backward_cache for the backward_pass.

    Args:
      ctx: information needed for the backward pass
      graph: the copy of the local graph
      score: logits-scores. weights on the edges, typically attention weights
      eids: edge ids to map the logits to. Default is ALL: to apply to all

    Returns:
      sparse_outputs: sparsemax applied to the logits-scores
    """

    if not is_all(eids):
      graph = graph.edge_subgraph(eids.long())
    # Store score in edge
    # assert False
    graph.edata['score'] = score
    graph.update_all(fn.copy_e('score', 'm'), fn.max('m', 'tmp'))
    graph.apply_edges(fn.e_sub_v('score', 'tmp', 'norm_score'))
    graph.edata.pop('score')
    graph.ndata.pop('tmp')
    graph.update_all(
        message_func=fn.copy_e('norm_score', 'norm_score'),
        reduce_func=reduce_tau_support_size)

    graph.apply_edges(edge_clamp)

    graph.edata.pop('norm_score')
    sparse_outputs = graph.edata.pop('out')
    support_size = graph.dstdata.pop('support_size')

    ctx.backward_cache = (graph, sparse_outputs, support_size)
    return sparse_outputs

  @staticmethod
  def backward(ctx, grad_out):
    """Backward pass of the Sparsemax operation.

    Args:
      ctx: information saved during the forward pass
      grad_out: gradients coming from next layers during backprop.

    Returns:
      requires one gradient for each of the parameter of the
      forward function

      None: no gradient for graph from forward
      grad_input: gradient for logits-scores from forward
      None: no gradient for eids from forward

    """
    graph, out, supp_size = ctx.backward_cache
    ctx.backward_cache = None

    grad_score = grad_out.clone()
    grad_score[out == 0] = 0

    graph.edata['grad_score'] = grad_score
    graph.dstdata['supp_size'] = supp_size

    # fast implementation using the built-in functions
    graph.update_all(
        fn.copy_e('grad_score', 'grad_score'),
        fn.sum('grad_score', 'v_hat'),
    )
    supp_size = supp_size.view(graph.ndata['v_hat'].shape)

    graph.ndata['v_hat'] = graph.ndata['v_hat'] / supp_size
    graph.apply_edges(
        fn.e_sub_v('grad_score', 'v_hat', 'grad_input_minus_vhat'))
    grad_input_minus_vhat = graph.edata.pop('grad_input_minus_vhat')
    grad_input = th.where(out != 0, grad_input_minus_vhat, grad_score)

    graph.dstdata.pop('v_hat')
    graph.dstdata.pop('supp_size')
    graph.edata.pop('grad_score')
    return None, grad_input, None


def edge_sparsemax(graph, logits, eids=ALL):
  """wrapper of edge sparsemax function.

  The message passing interface of dgl requires the wrapping
  of function using apply.

  Args:
    graph: the copy of the local graph
    logits: weights on the edges, typically attention weights
    eids: edge ids to map the logits to. Default is ALL: to apply to all nodes

  Returns:
    Results of EdgeSparsemax when applied to the given parameters
    using the member function apply.
  """

  return EdgeSparsemax.apply(graph, logits, eids)
