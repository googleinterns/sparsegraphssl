"""Performs empirical analysis on learned attention.

This module provides functionality to:
  - Measure the homophily degree on a dataset (intra-class-connectivity)
  - Distribution of attention weights (Entropy)
  - Noise robustness:
    - Injection of random connections:
      - random choices.
      - or to some samples from different classse
    - Detection of noisy connections at test time.
  - Reduced Branching between labeled nodes:
    - compute n-hops matrix between labeled nodes
    - take multiplication of these matrix.
    - Compare average, pair-wise length between intra-class nodes.
"""

import networkx as nx
import numpy as np
import torch
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import matplotlib.pyplot as plt
import os
from analysis.attention_vis import visualize_attentions
import random
from data_reader.data_utils import zip_edge
from tqdm import tqdm


def measure_homophily(graph, attention_weights, labels):
  """Measures homophily (intra-class) connectivity

  This func computes the average 1-hop connectivity between samples in the same
  class.
    - Sum over all intra-class connections
    - normalize by the number of nodes

  Args:
      graph: DGLGraph-graph with access to nodes and edges
      attention_weights: between 0-1
      labels: according to nodes indices

  Returns:
      mean_homophily: how many samples are connected within the same class
  """
  nodes = graph.nodes()
  edges = graph.edges()
  n_nodes = len(nodes)

  # transforms clasess to their classes.
  class_edges_src = labels[edges[0]]
  class_edges_trgt = labels[edges[1]]

  # find out which edges are in the same class
  homophily_edges = class_edges_src == class_edges_trgt
  # take only these edges and computes the weights
  all_weights_to_same_class = attention_weights[homophily_edges]

  mean_homophily = all_weights_to_same_class.sum() / n_nodes

  assert mean_homophily > 0,\
         'homophily is negative: {}'.format(mean_homophily)
  assert mean_homophily < 1,\
         'homophily is larger than 1: {}'.format(mean_homophily)
  return mean_homophily


def measure_entropy(graph, attention_weights):
  """Measures average entropy for each nodes.

  This function computes the entropy on each node using the message passing
  interface.
    - Entropy(node) = sum_{edges-i} p_i * log(p_i)

  Args:
      graph: DGLGraph-graph with access to nodes and edges
      attention_weights: between 0-1

  Returns:
      nodes_entropy: entropy value of all_nodes
  """

  weighted_p = -attention_weights * attention_weights.log()

  weighted_p[torch.isnan(weighted_p)] = 0

  assert weighted_p.shape == attention_weights.shape, \
         'Shapes of attention_weights and intermediate results should match'
  # sum using the message passing framework of dgl
  local_graph = graph.local_var()
  local_graph.edata['weighted_p'] = weighted_p
  local_graph.update_all(
      fn.copy_e('weighted_p', 'msg'),
      fn.sum('msg', 'nodes_entropy'),
  )
  nodes_entropy = local_graph.ndata.pop('nodes_entropy')
  assert len(nodes_entropy) == len(graph.nodes())
  mean_entropy = nodes_entropy.mean()

  return nodes_entropy, mean_entropy


def measure_ref_uniform_entropy(graph, attention_weights_shape):
  """Measures the entropy of the uniform distribution.

  Set all attention_weights to be equal. Normalize afterward

  Args:
      graph: DGLGraph-graph with access to nodes and edges
      attention_weights_shape: how are the attention_weights structured

  Returns:

  """

  uniform_attention = torch.ones(attention_weights_shape)
  uniform_attention_weights = edge_softmax(graph, uniform_attention)
  nodes_entropy, mean_entropy = measure_entropy(graph,
                                                uniform_attention_weights)

  return nodes_entropy, mean_entropy


def plot_attention_entropy(
    graph,
    attention_heads,
    dataset,
    samples_per_class,
    model_name,
    labels,
):
  """Plot attention entropy for each heads.

  This plot subdivides itself into subplots.
  N-Heads have n-subplots. The last subplot plots the random entropy.
  Title = Heads +  Diff to uniform_entropy

  Args:
      graph: DGLGraph-graph with access to nodes and edges
      attention_heads: dim 0: nodes, dim 1: heads
      dataset (str): Dataset name being used
      samples_per_class (int): how many samples/class to use
      model_name (str): which model is currently used
      labels (torch.LongTensor): Labels of all nodes in the dataest
  """
  save_path = 'vis_att/unsup/{}/{}'.format(dataset, samples_per_class)
  file_name = 'entropy_{}.png'.format(model_name)

  assert len(attention_heads.shape) == 2

  data = []
  for i in range(attention_heads.shape[1]):
    attention_weights = attention_heads[:, i]
    entropy, mean_entropy = measure_entropy(graph, attention_weights)
    homophily = measure_homophily(graph, attention_weights, labels)
    data.append((entropy, mean_entropy, homophily))

    # assert mean_entropy < 1.28

  uniform_entropy, uniform_mean_entropy = measure_ref_uniform_entropy(
      graph, attention_weights.shape)
  data.append((uniform_entropy, uniform_mean_entropy, None))

  os.makedirs(save_path, exist_ok=True)
  file_path = '{}/{}'.format(save_path, file_name)
  plot_and_save_histogram(
      data,
      file_path,
      attention_heads.shape[1],
      dataset,
      samples_per_class,
      model_name,
  )


def plot_and_save_histogram(
    data,
    file_path,
    n_heads,
    dataset,
    samples_per_class,
    model_name,
):
  """Plot and save the histogram.

  Args:
      data (tuple): stats for entropy, mean_entropy, homophily each.
      file_path (str): path to save the histogram'
      n_heads (int): how many attention heads to use.
      dataset (str): name of the dataset.
      samples_per_class (int): how many samples/class in training.
      model_name (str): Which model to be used.
  """

  subplots_horizontal = 3
  fig, axs = plt.subplots(
      int(np.ceil((n_heads + 1) / subplots_horizontal)),
      subplots_horizontal,
      figsize=(19.20, 10.80),
  )
  fig.suptitle(
      '{}/{}/{}'.format(dataset, samples_per_class, model_name), fontsize=16)
  uniform_data = data[-1]
  _, uniform_mean_entropy, _ = uniform_data
  for i, (entropy, mean_entropy, homophily) in enumerate(data):
    index_x = i // subplots_horizontal
    index_y = i % subplots_horizontal
    current_sub_plot = axs[index_x, index_y]
    entropy = entropy.detach().numpy()
    current_sub_plot.hist(entropy, facecolor='blue', alpha=0.75, bins=30)
    current_sub_plot.set_xlim([0, 4])
    entropy_diff = uniform_mean_entropy - mean_entropy
    if i < len(data) - 1:
      # n-heads
      title = 'Head:{} Diff-RandEnt: {:.2f}% Homophily {:.2f}%'.format(
          i, entropy_diff * 100, homophily * 100)
      assert entropy_diff > 0, 'Why is the diff to random distribution so small'
    else:
      # uniform weights
      title = 'Random:Uniform entropy {:.2f}'.format(uniform_mean_entropy)
    current_sub_plot.set_title(title)
  for ax in axs.flat:
    ax.set(xlabel='Entropy', ylabel='Number of nodes')
  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for ax in axs.flat:
    ax.label_outer()
  fig.savefig(file_path)


def collect_nodes(to_expand_nodes, edges):
  """Summary

  Args:
      to_expand_nodes (list): List of original nodes to expand from.
      edges (tuple): (src, trg). List of all edges in the graph.

  Returns:
      TYPE: Description
  """
  edges_src = edges[0]
  edges_trg = edges[1]
  all_new_nodes = set()
  orig_node_size = len(to_expand_nodes)
  for node in to_expand_nodes:
    new_nodes = set(edges_trg[(edges_src == node)].numpy())
    all_new_nodes.update(new_nodes)
  to_expand_nodes.update(all_new_nodes)
  new_node_size = len(to_expand_nodes)
  assert new_node_size >= orig_node_size, 'Reducing nodes not possible'
  return to_expand_nodes


def collect_edges(edges, extended_nodes):
  """Collect all edges based on the nodes to be extended.

  This function finds all involved edges and add them to a list

  Args:
      edges (list): all edges in the graph
      extended_nodes (list): which nodes to expand from.

  Returns:
      edges_ids: list of edges involved.
  """
  edges_src = edges[0]
  edges_trg = edges[1]
  final_node_mask = torch.zeros((edges_src.shape[0],))
  for node in extended_nodes:
    node_to_include = torch.logical_or(
        (edges_src == node),
        (edges_trg == node),
    )
    final_node_mask = torch.logical_or(final_node_mask, node_to_include)

  edges_ids = final_node_mask.nonzero()
  assert len(edges_ids) == final_node_mask.sum()
  return edges_ids


def visualize_graph(
    graph,
    att,
    labels,
    nodes_to_plot,
    edges_to_plot,
    dataset,
    model_name,
    samples_per_class,
):
  """Plot the graph and save it.

  Args:
      graph (nx.Graph): the graph to be plotted.
      att (torch.Tensor): Attention weights.
      labels (torch.LongTensor): Class labels of the nodes
      nodes_to_plot (list): which nodes to plot.
      edges_to_plot (list): which edges to plot.
      dataset (str): Name of the dataset
      model_name (str): Name of the model
      samples_per_class (int): How many samples to be used
  """

  fig, ax = plt.subplots(figsize=(19.20, 10.80))
  num_nodes = len(nodes_to_plot)
  gray = (.77,) * 3
  visualize_attentions(
      graph,
      att,
      ax=ax,
      nodes_labels=labels,
      nodes_to_plot=nodes_to_plot,
      edges_to_plot=edges_to_plot,
      last_color=gray,
  )
  ax.set_axis_off()
  sm = plt.cm.ScalarMappable(
      cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))

  sm.set_array([])

  plt.colorbar(sm, fraction=0.046, pad=0.01)
  dir_folder = 'vis_att/all_classes/{}/{}'.format(
      dataset,
      samples_per_class,
  )
  os.makedirs(dir_folder, exist_ok=True)
  fig.savefig(
      '{}/unlabeled_{}_{}.png'.format(dir_folder, model_name, num_nodes),
      dpi=100)


def plot_unlabeled_samples(
    graph,
    attention_weights,
    labels,
    train_mask,
    dataset: str,
    model_name: str,
    samples_per_class: int,
    n_hops: int = 1,
) -> None:
  """Plots the attentions weights of training samples_only.

  Collect nodes of several_hops away. Simply looping over existing set
  of nodes. Visualize the graph based on these nodes.
  Labeled nodes should be colored according to their class.
  Unlabeled nodes should get extra-color (Gray)

  Args:
      graph: [DGLGraph] DGLGraph
      attention_weights: [] attention_weights of one head
      labels: [BoolTensor] Correct labels class
      train_mask: [BoolTensor] which samples used for training
      dataset (str): : Name of the dataset
      model_name (str): model name
      samples_per_class (int):  Samples/class used for training. Returns
      n_hops (int, optional): Description
  """

  LOWER_BOUND = 1e-3
  nodes = graph.nodes()
  edges = graph.edges()
  n_edges = len(edges[0])

  to_expand_nodes = set(nodes[train_mask].numpy())
  for hop in range(n_hops):
    to_expand_nodes = collect_nodes(to_expand_nodes, edges)
    print('Hop: {} total:{}'.format(hop, len(to_expand_nodes)))
    assert len(to_expand_nodes) > train_mask.sum()

  edges_ids = collect_edges(edges, to_expand_nodes)

  reduced_att = attention_weights[edges_ids].squeeze().numpy()
  reduced_edges = (edges[0][edges_ids], edges[1][edges_ids])

  att_edge_mask = attention_weights > LOWER_BOUND
  edges_mask = torch.zeros((n_edges,)).bool()
  edges_mask[edges_ids] = True
  reduced_att_ids = torch.logical_and(edges_mask, att_edge_mask).nonzero()
  edges_to_plot = zip_edge(
      (edges[0][reduced_att_ids], edges[1][reduced_att_ids]))

  reduced_edges = zip_edge(reduced_edges)
  reduced_graph = nx.Graph(reduced_edges)
  nodes_to_plot = sorted(reduced_graph.nodes())
  assert len(nodes_to_plot) >= len(to_expand_nodes)
  nodes = torch.LongTensor(nodes_to_plot)
  reduced_labels = labels[nodes]
  # extra color for all unlabeled samples
  reduced_labels[train_mask[nodes].logical_not()] = reduced_labels.max() + 1

  assert len(reduced_labels) >= len(to_expand_nodes)
  assert len(nodes) >= len(to_expand_nodes)
  assert len(reduced_labels) == len(nodes)
  assert len(edges_to_plot) <= len(reduced_edges)
  assert len(reduced_att_ids) <= len(edges_ids)
  assert len(nodes_to_plot) > 3

  visualize_graph(
      reduced_graph,
      reduced_att,
      reduced_labels,
      nodes_to_plot,
      edges_to_plot,
      dataset,
      model_name,
      samples_per_class,
      largest_connected_graph=True,
      full_edges=edges,
  )


def analyze_noise_detection(
    attention_weights,
    noisy_edges_ids,
    n_nodes,
):
  """Anayze the results of detecting noise with atteniton.

  Att on noisy edges/ nodes.

  Args:
      attention_weights: vector of attention_weights
      noisy_edges_ids: which ids are noisy. Should be starting from a large
        numbers toch.array
      n_nodes: how many nodes?
  No Longer Returned:
      True-Positive-Rate: how many were correctly identified
      False-Positive-Rate: how many were missed
  """
  # reduce the noise a little
  attention_weights = attention_weights.squeeze()
  assert attention_weights.ndim == 1

  weights_on_wrong_edges = attention_weights[noisy_edges_ids].sum().item(
  ) / n_nodes
  weights_on_wrong_edges = round(weights_on_wrong_edges * 100, 2)
  print('Mean-Att-Wrong-Edge: %.2f' % weights_on_wrong_edges)
  return weights_on_wrong_edges


def set_seed(seed=None):
  """Set all seeds to remove randomness

  This function sets all seeds to make experiments deterministic.

  Args:
      seed (None, optional): a seed to be used.
  """
  if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def compute_att_on_shortest_paths(
    graph: nx.Graph,
    attention_weights: torch.FloatTensor,
    labels: torch.LongTensor,
):
  """Computes how much attention is spent on the shortest simple_paths.

  Compute pair-wise shortest paths between all nodes. Based on these paths,
  compute multiplicative attention. Higher attention means better model.
  Only measure influences from samples within the same class.

  Args:
      graph (nx.Graph): Description
      attention_weights (torch.FloatTensor): weights between 0 and 1 for all
        edges.
      labels (torch.LongTensor): needs to separate samples into the same class
        graph (nx.Graph)
  """

  # from nx.documentation: If neither the source nor target are specified return a dictionary of dictionaries with path[source][target]=[list of nodes in path]

  edges = graph.edges()
  nx_graph = graph.cpu().to_networkx()
  if type(labels) == torch.Tensor:
    labels = labels.data.cpu().numpy()

  # convert attention to a dictionary keyed by edges for quick access.
  attention_dict = {
      (u, v): att for (u, v), att in zip(zip_edge(edges), attention_weights)
  }

  paths_dict = nx.shortest_path(nx_graph)
  # Find out which nodes belong together
  node_lists_per_class = [
      np.nonzero(labels == cl)[0].squeeze() for cl in range(labels.max() + 1)
  ]

  # Now iterate over all node pairs.
  all_weights: list = []
  for node_list in node_lists_per_class:
    all_pair_combination = np.meshgrid(node_list, node_list)
    u = all_pair_combination[0].flatten()
    v = all_pair_combination[1].flatten()
    all_weights_per_class = []

    for src, trg in tqdm(zip(u, v)):
      if src != trg and trg in paths_dict[src]:
        path = paths_dict[src][trg]
        att_pairs = zip(path, path[1:])
        att_score = [attention_dict[(u, v)] for u, v in att_pairs]
        att_mean_score = np.prod(att_score).item()
        all_weights_per_class.append(att_mean_score)
        assert len(att_score) > 0
    att_mean_per_class = np.mean(all_weights_per_class)
    all_weights.append(att_mean_per_class)
    print('Paths in this class %d' % len(all_weights_per_class))
  assert len(all_weights) > 0, 'Where are the elements %d ' \
                               % len(all_weights)
  att_mean = np.mean(all_weights).item()
  print('Att_mean on shortest paths %f' % att_mean)
  return att_mean
