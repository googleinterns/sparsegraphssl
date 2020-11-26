"""Collection of attentional models in DGL."""
import torch.nn as nn
import torch
from model_archs.gnn_convs.custom_gat_conv import CustomGATConv
from model_archs.gnn_convs.sparse_gat_conv import SparseGATConv
from model_archs.gnn_convs.label_prop_gat_conv import LabelPropGATConv
from model_archs.gnn_convs.label_prop_sparse_gat_conv import LabelPropSparseGATConv
from model_archs.gnn_convs.pooling import Pooling
import pdb


def extract_attention_weights(graph_layers, layer):
  """Extracts weights from layers.

  This function wraps the extraction of attention weights from all layers.

  Args:
      graph_layers (list of nn.Module): all graph layer.
      layer (int): which layers to pick for extraction.

  Returns:
      tensor: attention weights.
  """
  attention_weights = graph_layers[layer].attention_weights
  assert len(attention_weights.shape) == 3, 'make sure the format is correct.'
  attention_weights = attention_weights.mean(1)
  return attention_weights


class GAT(nn.Module):
  """GAT-model.

  Attributes:
      activation (activation): activation fucntion.
      g (DGLGraph): graph.
      graph_layers (list): list of graph layers.
      num_layers (int): how many layer for the model.
  """

  def __init__(self,
               g,
               num_layers,
               in_dim,
               num_hidden,
               num_classes,
               heads,
               activation,
               feat_drop,
               attn_drop,
               negative_slope,
               residual,
               conv=CustomGATConv,
               **kwargs):
    """Initialize the model.

    Args:
        g (DGLGraph): graph.
        num_layers (int): how many layer.
        in_dim (int): Description.
        num_hidden (int): number of hidden units.
        num_classes (int): Number of clasess.
        heads (int): How many heads for attention.
        activation (activation): activation fucntion..
        feat_drop (float): how much dropout to be used in the respective layer.
        attn_drop (float): how much dropout to be used in the respective layer.
        negative_slope (float): negative_slope of the activation function.
        residual (bool): Use residual connection or not.
        conv (TYPE, optional): which convolutional type to be used.
        **kwargs: all other keyword-arguments.
    """
    assert num_layers >= 1, 'GAT support only min. 2 Layers'
    super(GAT, self).__init__()
    self.g = g
    self.num_layers = num_layers
    self.graph_layers = nn.ModuleList()
    self.activation = activation
    conv_layer = conv(in_dim, num_hidden, heads[0], feat_drop, attn_drop,
                      negative_slope, False, self.activation, **kwargs)
    self.graph_layers.append(conv_layer)

    # hidden layers
    for l in range(1, self.num_layers):
      # due to multi-head, the in_feats = num_hidden * num_heads
      self.graph_layers.append(
          conv(num_hidden * heads[l - 1], num_hidden, heads[l], feat_drop,
               attn_drop, negative_slope, residual, self.activation, **kwargs))
    last_conv_layer = conv(num_hidden * heads[-2], num_classes, heads[-1],
                           **kwargs)
    self.graph_layers.append(last_conv_layer)

  def forward(self, inputs):
    """Computes the forward pass.

    Args:
        inputs (tensor): input features.

    Returns:
        logits: output of the forward passes
    """
    h = inputs
    for layer in range(self.num_layers):
      h = self.graph_layers[layer](self.g, h).flatten(1)
    logits = self.graph_layers[-1](self.g, h).mean(1)
    return logits


class SparseGAT(GAT):
  """GAT model with SparseMax atteniton

  Attributes:
      lambda_sparsemax (float): temperature of sparsemax.
  """

  def __init__(self, *args, lambda_sparsemax=None, **kwargs):
    """Initialize the model.

    Args:
        *args: params for the parent model GAT
        lambda_sparsemax (None, optional): temperature of sparsemax.
        **kwargs: other required params for SparseMax.
    """
    super(SparseGAT, self).__init__(
        *args, conv=SparseGATConv, lambda_sparsemax=lambda_sparsemax, **kwargs)
    self.lambda_sparsemax = lambda_sparsemax


class LabelPropGAT(GAT):
  """GAT with Label-propagation.

  Attributes:
      label_prop_steps (int): how many steps for LP.
      mean_pooling_layer (layer): a mean-pooling-layer.
      use_adj_matrix (bool): Use adjacency matrix or attention matrix.
      weighted_pooling_layer (layer): Adj-matrix requires this layer to do LP.
  """

  def __init__(self,
               *args,
               pooling_residual=None,
               use_adj_matrix=None,
               label_prop_steps=None,
               conv=LabelPropGATConv,
               **kwargs):
    """Initialize the LabelPropGAT

    Args:
        *args: params for the parent model GAT
        pooling_residual (None, optional): use residual or not.
        use_adj_matrix (None, optional):  Use adjacency matrix or attention
          matrix.
        label_prop_steps (None, optional):  how many steps for LP.
        conv (TYPE, optional): Which conv layer to be used.
        **kwargs: other required params for SparseMax.
    """
    super(LabelPropGAT, self).__init__(*args, conv=conv, **kwargs)
    num_classes = args[4]
    self.label_prop_steps = label_prop_steps
    self.weighted_pooling_layer = Pooling(
        num_classes,
        num_classes,
        aggregator_type='weighted_pooling',
        feat_drop=0,
        activation=None,
        pooling_residual=pooling_residual,
    )
    self.mean_pooling_layer = Pooling(
        num_classes,
        num_classes,
        aggregator_type='mean',
        feat_drop=0,
        activation=None)
    self.use_adj_matrix = use_adj_matrix

  def forward(self, inputs):
    """Computes the forward pass.

    Args:
        inputs (tensor): input features.

    Returns:
        logits: output of the forward passes
    """
    attention_weights_list = []
    h = inputs
    for layer in range(self.num_layers):
      h = self.graph_layers[layer](self.g, h).flatten(1)
      extracted_weights = extract_attention_weights(self.graph_layers, layer)
      attention_weights_list.append(extracted_weights)
    logits = self.graph_layers[-1](self.g, h).mean(1)

    extracted_weights = extract_attention_weights(self.graph_layers, -1)
    attention_weights_list.append(extracted_weights)

    reduce_dim = -1
    all_attentions_weights = torch.stack(attention_weights_list, reduce_dim)
    mean_attention_weights = all_attentions_weights.mean(reduce_dim).detach()
    pseudo_labels = torch.softmax(logits, dim=1)

    pseudo_labels_list = [pseudo_labels]
    for i in range(self.label_prop_steps):
      if self.use_adj_matrix:
        logits = self.mean_pooling_layer(self.g, pseudo_labels)
      else:
        logits = self.weighted_pooling_layer(self.g, pseudo_labels,
                                             mean_attention_weights)

      pseudo_labels = torch.softmax(logits, dim=1)
      pseudo_labels_list.append(pseudo_labels)
      output_pseudo_labels = pseudo_labels_list[-1]

    output_pseudo_labels = pseudo_labels_list[-1]
    log_softmax_labels = torch.log(output_pseudo_labels)
    return log_softmax_labels


class LabelPropGATMultilabelFinal(LabelPropGAT):
  """LP-GAT for multi-label-learning."""

  def forward(self, inputs):
    """Computes the forward pass.

    Args:
        inputs (tensor): input features.

    Returns:
        logits: output of the forward passes
    """
    attention_weights_list = []
    h = inputs
    for layer in range(self.num_layers):
      h = self.graph_layers[layer](self.g, h).flatten(1)
      extracted_weights = extract_attention_weights(self.graph_layers, layer)
      attention_weights_list.append(extracted_weights)

    logits = self.graph_layers[-1](self.g, h).mean(1)
    extracted_weights = extract_attention_weights(self.graph_layers, -1)
    attention_weights_list.append(extracted_weights)
    mean_attention_weights = torch.stack(attention_weights_list, -1).mean(-1)
    for i in range(self.label_prop_steps):
      if self.use_adj_matrix:
        logits = self.mean_pooling_layer(self.g, logits)
      else:
        logits = self.weighted_pooling_layer(self.g, logits,
                                             mean_attention_weights)
    return logits


class LabelPropSparseGAT(LabelPropGAT):
  """SparseGAT with LabelPropagation."""

  def __init__(self, *args, **kwargs):
    """Initialize the class with LabelPropSparseGATConv.

    Args:
        *args: all required args for parent class.
        **kwargs: all required args for parent class.
    """
    super(LabelPropSparseGAT, self).__init__(
        *args, conv=LabelPropSparseGATConv, **kwargs)
