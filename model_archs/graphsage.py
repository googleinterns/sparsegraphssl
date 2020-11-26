"""GraphSAGE-Model
"""
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn as nn


class GraphSAGE(nn.Module):

  """GraphSAGE-model.
  
  Attributes:
      graph_layers (list): list of all graph layers
  """
  
  def __init__(self, g, in_feats, n_hidden, n_classes, num_layers, activation,
               dropout, aggregator_type):
    """Initialize the GraphSAGE model
    
    Args:
        g (DGLGraph): graph.
        in_feats (int): number of input features.
        n_hidden (int): numbers of hidden units.
        n_classes (int): how many classes.
        num_layers (int): how mnay layers to use.
        activation (activation): activation type.
        dropout (float): how much dropout?
        aggregator_type (aggregator_type): type of aggregatation layer.
    """
    super(GraphSAGE, self).__init__()
    self.graph_layers = nn.ModuleList()
    # input layer
    self.graph_layers.append(
        SAGEConv(
            in_feats,
            n_hidden,
            aggregator_type,
            feat_drop=dropout,
            activation=activation))
    # hidden layers
    for i in range(num_layers - 1):
      self.graph_layers.append(
          SAGEConv(
              n_hidden,
              n_hidden,
              aggregator_type,
              feat_drop=dropout,
              activation=activation))
    # output layer
    self.graph_layers.append(
        SAGEConv(
            n_hidden,
            n_classes,
            aggregator_type,
            feat_drop=dropout,
            activation=None))

  def forward(self, features):
     """Computes the forward pass
    
    Args:
        features (tensor): input features of all nodes.
    
    Returns:
        output: output of forward-pass through the model.
    """
    h = features
    for layer in self.graph_layers:
      h = layer(self.g, h)
    return h
