"""GCN from DGL."""
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
  """Summary

  Attributes:
      dropout (float): how much dropout?
      g (DGLGraph): graph.
      layers (TYPE): container of layers.
  """

  def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation,
               dropout):
    """Summary

    Args:
        g (DGLGraph): graph.
        in_feats (int): number of input features.
        n_hidden (int): numbers of hidden units.
        n_classes (int): how many classes.
        n_layers (int): how many layers.
        activation (activation): activation type.
        dropout (float): how much dropout?
    """
    super(GCN, self).__init__()
    self.g = g
    self.layers = nn.ModuleList()
    # input layer
    self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
    # hidden layers
    for i in range(n_layers - 1):
      self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
    # output layer
    self.layers.append(GraphConv(n_hidden, n_classes))
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, features):
    """Computes the forward pass

    Args:
        features (tensor): input features of all nodes.

    Returns:
        output: output of forward-pass through the model.
    """
    h = features
    for i, layer in enumerate(self.layers):
      if i != 0:
        h = self.dropout(h)
      h = layer(self.g, h)
    return h
