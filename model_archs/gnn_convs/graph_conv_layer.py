"""GraphConvLayer for Jknet."""
# Lint as  python3

import dgl.function as fn
from frozendict import frozendict
import torch

AGGREGATIONS = frozendict({
    'sum': torch.sum,
    'mean': torch.mean,
    'max': torch.max,
})


class GraphConvLayer(torch.nn.Module):
  """Graph convolution layer for Jknet.

  Compute the graph convolutional operation within the JKNET.

  """

  def __init__(self, in_features, out_features, aggregation='sum'):
    """Graph convolution layer for Jknet.

    Compute the graph convolutional operation within the JKNET.

    Args:
      in_features (int): Size of each input node.
      out_features (int): Size of each output node.
      aggregation (str): 'sum', 'mean' or 'max'. Specify the way to aggregate
        the neighbourhoods.
    """

    super(GraphConvLayer, self).__init__()

    if aggregation not in AGGREGATIONS.keys():
      raise ValueError("'aggregation' {} is not one of  "
                       "'sum', 'mean' or 'max'.".format(aggregation))
    self.aggregate = lambda nodes: AGGREGATIONS[aggregation](nodes, dim=1)

    self.linear = torch.nn.Linear(in_features, out_features)
    self.self_loop_w = torch.nn.Linear(in_features, out_features)
    self.bias = torch.nn.Parameter(torch.zeros(out_features))

  def forward(self, graph, input_feat):
    """Implements forward pass of the Graph convolution layer in Jknet.

    Compute the forward pass of graph convolutional operation within the JKNET.

    Args:
      graph: local copy of the graph
      input_feat: input features

    Returns:
      results: torch Tensor after applying convolution.
    """
    graph.ndata['hidden_feat'] = input_feat
    graph.update_all(
        fn.copy_src(src='hidden_feat', out='msg'),
        lambda nodes: {'hidden_feat': self.aggregate(nodes.mailbox['msg'])})
    hidden_feat = graph.ndata.pop('hidden_feat')
    hidden_feat = self.linear(hidden_feat)
    return hidden_feat + self.self_loop_w(input_feat) + self.bias
