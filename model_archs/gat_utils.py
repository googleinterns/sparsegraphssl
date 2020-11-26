"""Utils with some functions to create models."""
import torch.nn.functional as F
import torch
import numpy as np
# from model_archs.graphsage import GraphSAGE
# from model_archs.gcn import GCN

import model_archs.gat_models as gat_models_lib
from dgl import function as fn
from data_reader.data_utils import add_noise_to_graph
import torch.nn as nn


def build_model(g, args, num_feats, n_classes, heads):
  """Builds models based on arguments.

  Args:
      g (DGLGraph): graph.
      args (TYPE): Description
      num_feats (int): how many features as inputs.
      n_classes (int): how many classes.
      heads (int: how many attention_heads

  Returns:
      model: fully-initialized model.

  Raises:
      NotImplementedError: some models are not yet implemented.
  """
  if 'GAT' not in args.model:
    if args.data.startswith('ppi'):
      args.num_hidden = 512
    elif args.data.startswith('yelp'):
      args.num_hidden = 128
    else:
      args.num_hidden = 16

  model_type = args.model
  if 'GAT' in model_type:
    gat_model = getattr(gat_models_lib, model_type)
    model = gat_model(
        g,
        args.num_layers,
        num_feats,
        args.num_hidden,
        n_classes,
        heads,
        F.elu,
        args.in_drop,
        args.attn_drop,
        args.alpha,
        args.residual,
        lambda_sparsemax=args.lambda_sparsemax,
        label_prop_steps=args.label_prop_steps,
    )
  elif 'GCN' in model_type:
    g = normalize_graph(g, args.gpu)

    model = GCN(g, num_feats, args.num_hidden, n_classes, args.num_layers,
                F.relu, args.dropout)
    model.graph_layers = model.layers
  elif 'GraphSAGE' in model_type:
    # create GraphSAGE model
    model = GraphSAGE(
        g,
        num_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
        aggregator_type=args.graphsage_aggregator_type)
  else:
    print(model_type)
    raise NotImplementedError
  return model


def normalize_graph(g, gpu):
  """Normalizes with into laplacian matrix.

  Args:
      g (DGLGraph): which graph to be used.
      gpu (int): which gpu for use.

  Returns:
      graph: graph
  """
  degs = g.in_degrees().float()
  norm = torch.pow(degs, -0.5)
  norm[torch.isinf(norm)] = 0
  if gpu >= 0:
    norm = norm.cuda()
  g.ndata['norm'] = norm.unsqueeze(1)
  return g


def compute_sparsity_stats(attention_weights_list):
  """Anayzes the sparsity of learned graphs.

  Args:
      Attention_weights_list (list): list of attention_weights

  Returns:
      sparsity_percentage: how many entries were cut off by sparse-attention
  """
  all_zeros_entries = []
  all_elems = []

  for att in attention_weights_list:
    n_zeroes_entries = (att == 0).sum()
    all_zeros_entries.append(n_zeroes_entries)
    all_elems.append(torch.numel(att))
  total_number_sparse_att = np.sum(np.array(all_zeros_entries))
  total_number_att = np.sum(np.array(all_elems))

  sparsity_percentage = float(total_number_sparse_att) / total_number_att
  print(
      'Sparsity percent: {:.2f} total_number sparse att {}  total_number_att {}'
      .format(sparsity_percentage * 100, total_number_sparse_att,
              total_number_att))

  return sparsity_percentage


def collect_attention_weights(model, mean=False, keep_list=False):
  """Collect all attentions weights into a tensor

  Args:
      model (model): the graph model.
      mean (bool, optional): Take the mean only?
      keep_list (bool, optional): Keep the entire list of weights?

  Returns:
      TYPE: Description
  """
  all_weights = []
  for layer in model.graph_layers:
    att_w = layer.attention_weights
    if len(att_w.shape) == 1:
      # single head attention_weights
      att_w = att_w.view(-1, 1)
    if mean:
      att_w = att_w.mean(1).unsqueeze(1)
    all_weights.append(att_w)

  assert att_w.shape[1] >= 1
  if not keep_list:
    att_w_tensors = torch.cat(all_weights, dim=1).squeeze()
    att_w_tensors = att_w_tensors.mean(1)
    return att_w_tensors
  else:
    squeezed_weights = [w.squeeze() for w in all_weights]
    return squeezed_weights


def compute_tree_sparsification(graph, attention_list, init_neighbors=None):
  """Compute the average sparsification of the interpretability tree.

  Finds out which nodes have the largest interpretability introduction.
  Also computes how much of trees are removed in average.

  Args:
      graph: DGLGraph. Use message passing to compuet how many nodes are needed
        in the tree.
      attention_list: List of attention_weights. Last layer = last element.
      init_neighbors (None, optional): The neighbors-count to be used.
  Return:
      all_neighbors_count: percentage of saved neighborhood
  """
  local_graph = graph.local_var()
  n_nodes = len(graph.nodes())
  if init_neighbors is None:
    init_neighbors = torch.ones((n_nodes,)).type_as(attention_list[0])
  local_graph.ndata['neighbors'] = init_neighbors
  for att_weight in attention_list:
    # perform message passing to aggregate nodes
    local_graph.edata['att'] = att_weight
    local_graph.apply_edges(fn.e_mul_u('att', 'neighbors', 'counted_neighbors'))
    local_graph.update_all(
        fn.copy_e('counted_neighbors', 'counted_neighbors'),
        fn.sum('counted_neighbors', 'neighbors'))
    all_neighbors_count = local_graph.ndata['neighbors']
  return all_neighbors_count


def compare_tree_sparsification(graph, attention_list):
  """Find out which nodes are sparsitfied the most.

  Performs deep analysis of sparsemax after learning.

  Computes:
        - mass of the interpretability tree is reduced in expectation and which
        node
        is the best to display in term of sparsification
        - attention_mass placed on same labels-examples.
      Args: graph attention_list

  Args:
      graph (DGLGraph): the graph.
      attention_list (list): list of attentions.
  Return:
      saving: vector of how many nodes are saved in %.
      average_saving: how much of the interpretability tree is cutoff
      most_sparse_node: Which node should be plotted
  """
  max_neighbor_for_plot = 40
  min_neighbor_for_plot = 5
  random_attention_list = [att * 0 + 1 for att in attention_list]
  ref_neighbor_counts = compute_tree_sparsification(graph,
                                                    random_attention_list)

  clamped_attention_list = [(att != 0).float() for att in attention_list]
  real_neighbor_counts = compute_tree_sparsification(graph,
                                                     clamped_attention_list)
  saving = 1 - (real_neighbor_counts / ref_neighbor_counts)
  # convert into percent
  saving = saving * 100
  average_saving = saving.mean()
  saving[real_neighbor_counts > max_neighbor_for_plot] = 0
  saving[real_neighbor_counts < min_neighbor_for_plot] = 0
  most_sparse_node = torch.argmax(saving)
  assert saving[most_sparse_node] > 0
  # pdb.set_trace()
  return saving, average_saving, most_sparse_node


def compute_att_on_noisy_edges(graph, model, features, labels):
  """Computes attentions-mass placed on clean/noisy connections.

  Suppress the noisy edges with attention = 0 after model-forward to see how
  much attention mass is placed on the clean connections.

  Args:
      graph (DGLGraph): the graph to be analyzed
      model (model): the graph model to be used.
      features (tensor): features from all nodes
      labels (tensor): Labels of all nodes.
  Deleted Parameters:
      Return: results  of noise testing analyses.

  Returns:
      TYPE: Description
  """

  noise_ratios = [.1, .2, .5]
  noise_types = ['random', 'heterophily']
  results = {}
  for noise_type in noise_types:
    results[noise_type] = {}
    for noise_ratio in noise_ratios:
      noisy_graph, noisy_edge_ids = add_noise_to_graph(
          graph,
          noise_ratio=noise_ratio,
          noise_type=noise_type,
          labels=labels,
      )
      # collect the attention_weights
      model.eval()
      model.g = noisy_graph
      _ = model(features)
      attention_list = collect_attention_weights(
          model,
          mean=True,
          keep_list=True,
      )
      # modify the attention_list to set att on noisy edges to 0
      clean_mask = torch.ones((noisy_graph.num_edges(),))
      clean_mask[noisy_edge_ids] = 0
      clean_mask = clean_mask.type_as(labels)
      clean_attn = [att * clean_mask for att in attention_list]
      mass_on_clean_nodes = compute_tree_sparsification(noisy_graph, clean_attn)
      mean_on_clean_nodes = mass_on_clean_nodes.mean() * 100

      print('Type {} ratio {} correct_att {:.2f}'.format(
          noise_type,
          noise_ratio,
          mean_on_clean_nodes,
      ))
      results[noise_type][noise_ratio] = mean_on_clean_nodes
  return results


def compute_att_mass(graph, attention_list, init_neighbors=None):
  """Compute the average sparsification of the interpretability tree.

  Finds out which nodes have the largest interpretability introduction.
  Also computes how much of trees are removed in average.

  Args:
      graph: DGLGraph. Use message passing to compuet how many nodes are needed
        in the tree.
      attention_list: List of attention_weights. Last layer = last element.
      init_neighbors (None, optional): Description
  Return:
      all_neighbors_count: percentage of saved neighborhood
  """
  local_graph = graph.local_var()
  n_nodes = len(graph.nodes())
  if init_neighbors is None:
    init_neighbors = torch.ones((n_nodes,)).type_as(attention_list[0])
  local_graph.ndata['init_neighbors'] = init_neighbors
  local_graph.ndata['neighbors'] = torch.ones(
      (n_nodes,)).type_as(attention_list[0])
  for att_weight in attention_list:
    # perform message passing to aggregate nodes
    # sum_{all edges} (att_weight * src['neighbors']])
    local_graph.edata['att'] = att_weight
    local_graph.apply_edges(fn.e_mul_u('att', 'neighbors', 'counted_neighbors'))
    local_graph.apply_edges(
        fn.e_mul_u('counted_neighbors', 'init_neighbors', 'att_mass'))
    local_graph.update_all(
        fn.copy_e('att_mass', 'att_mass'), fn.sum('att_mass', 'neighbors'))
    all_neighbors_count = local_graph.ndata['neighbors']
  return all_neighbors_count


def compute_att_on_same_class(graph, model, features, labels):
  """Computes attentions-mass placed on samples_from the same_class

  Suppress the noisy edges with attention = 0 after model-forward to see how
  much attention mass is placed on the clean connections. Compute the density
  additionally (on how many samples are the attentions spread to)

  Args:
      graph (DGLGraph): the graph to be analyzed
      model (model): the graph model to be used.
      features (tensor): features from all nodes
      labels (tensor): Labels of all nodes.

  Returns:
      resutls: analyzed results.
  """

  noise_ratios = [0, .1, .2, .3, .5, .8, 1, 2]
  noise_types = ['heterophily']
  # noise_types = ['random']
  results = {}
  for noise_type in noise_types:
    results[noise_type] = {}
    for noise_ratio in noise_ratios:
      noisy_graph, noisy_edge_ids = add_noise_to_graph(
          graph,
          noise_ratio=noise_ratio,
          noise_type=noise_type,
          labels=labels,
      )
      # collect the attention_weights
      model.eval()
      model.g = noisy_graph
      _ = model(features)
      attention_list = collect_attention_weights(
          model,
          mean=True,
          keep_list=True,
      )
      init_neighbors = nn.functional.one_hot(labels).type_as(features)
      att_on_same_class = compute_att_mass(
          noisy_graph,
          attention_list,
          init_neighbors,
      )
      assert att_on_same_class.max() > .1, 'Atts are too small %f. Why?'\
                                           % att_on_same_class.max()
      # compute mass for each class separately:
      mean_att = att_on_same_class.sum() / graph.num_nodes()
      mean_att = mean_att * 100
      assert mean_att > 0, 'Why so small mean att % f' % mean_att

      clamped_attention_list = [(att != 0).float() for att in attention_list]
      involved_samples_for_att = compute_att_mass(
          noisy_graph,
          clamped_attention_list,
          init_neighbors,
      )
      att_density = (att_on_same_class * 100 / involved_samples_for_att)
      att_density[torch.isnan(att_density)] = 0
      att_density_mean = att_density.sum() / graph.num_nodes()
      print('Type {} ratio {} att_density_mean {:.2f}'.format(
          noise_type,
          noise_ratio,
          att_density_mean,
      ))

      results[noise_type][noise_ratio] = (mean_att, att_density_mean)
  return results
