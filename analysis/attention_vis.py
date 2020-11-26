"""Visualize the learned attentions.

This module visualizes the attentions in the following forms:
  - Standard: All nodes are colored according to their class.
    Sample n-nodes from the full graph. Take the largest,
    connecting set of graph. Plot attentions edge weight.
  - SSL: Treat unlabeled labels as grey. Labeled samples have their color.
    Plot n-hops neighbors of these samples.


"""

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import os
from networkx.drawing.nx_agraph import graphviz_layout
import torch
import dgl


def visualize_attentions(
    g,
    attention,
    ax,
    nodes_to_plot=None,
    nodes_labels=None,
    edges_to_plot=None,
    nodes_colors=None,
    edge_colormap=plt.cm.Reds,
    log_scale=False,
    last_color=None,
):
  """Visualize edge attentions by coloring edges on the graph.

  Plots the graph with different options.

  Args:
      g: nx.DiGraph Directed networkx graph
      attention: list Attention values corresponding to the order of
        sorted(g.edges())
      ax: matplotlib.axes._subplots.AxesSubplot ax to be used for plot
      nodes_to_plot: list List of node ids specifying which nodes to plot.
        Default to be None. If None, all nodes will be plot.
      nodes_labels: list, numpy.array nodes_labels[i] specifies the label of the
        ith node, which will decide the node color on the plot. Default to be
        None. If None, all nodes will have the same canonical label. The
        nodes_labels should contain labels for all nodes to be plot.
      edges_to_plot: list of 2-tuples (i, j) List of edges represented as
        (source, destination). Default to be None. If None, all edges will be
        plot.
      nodes_colors: list Specifies node color for each node class. Its length
        should be bigger than number of node classes in nodes_labels.
      edge_colormap: plt.cm Specifies the colormap to be used for coloring
        edges.
      log_scale (bool, optional): Use log-scale for plotting?
      last_color (None, optional): Which color to use for the last class.
  """
  if nodes_to_plot is None:
    nodes_to_plot = sorted(g.nodes())
  if edges_to_plot is None:
    assert isinstance(g, nx.DiGraph), 'Expected g to be an networkx.DiGraph' \
                                      'object, got {}.'.format(type(g))
    edges_to_plot = sorted(g.edges())
  nodes_pos = nx.spring_layout(g)
  if log_scale:
    v_min = -100
    log_attention_weights = np.log(attention)
    log_attention_weights[attention == 0] = v_min
    log_attention_weights /= 100 + 1
  else:
    v_min = 0
    v_max = 1

  nx.draw_networkx_edges(
      g,
      nodes_pos,
      edgelist=edges_to_plot,
      edge_color=attention,
      edge_cmap=edge_colormap,
      width=attention * 10,
      alpha=0.5,
      ax=ax,
      edge_vmin=0,
      edge_vmax=1)
  if nodes_colors is None:
    nodes_colors = sns.color_palette('deep', max(nodes_labels) + 1)
    if last_color:
      nodes_colors[-1] = last_color
    colors = [
        nodes_colors[nodes_labels[i]] for i, v in enumerate(nodes_to_plot)
    ]
    degree = nx.degree(g)
    node_size = [v * 10 for v in dict(degree).values()]
    nx.draw_networkx_nodes(
        g,
        nodes_pos,
        nodelist=nodes_to_plot,
        ax=ax,
        node_color=colors,
        alpha=0.9,
        node_size=node_size)


def to_args(all_edges, edges):
  """Convert edges to edges_ids

  Args:
      all_edges (list): All edges in graphs.
      edges (list): the query edges.

  Returns:
      list: The ids of the edges.
  """
  all_indices = []
  for elem in edges:
    index = all_edges.index(elem)
    all_indices.append(index)
  return np.array(all_indices)


def plot_attentions(g, model, labels, args):
  """Plots atteniton based on attention_weights


  Visualize the plot of attention. Should be used after training and without
  attention dropout.
  Plot the test nodes and their neighbors only

  Args:
      g: DGLGraph. Needs to convert to nx.DiGraph
      model (str): name of the model.
      labels (torch.LongTensor): Labels of all nodes in the graph
      args (namespace): All hyperparameter of experiments
  Deleted Parameters:
      attention_weights: list of atetntion weights
  """
  fig, ax = plt.subplots(figsize=(19.20, 10.80))

  in_degrees = g.in_degrees()
  sorted_degrees = in_degrees.argsort(descending=True)

  n_nodes_to_plot = args.n_nodes_to_plot
  nodes = sorted_degrees[:n_nodes_to_plot]

  edges_to_plot = []
  edges_for_graphs = []
  nodes_to_plot = set()
  edges_ids = []
  lower_bound = 1e-3
  attention_weights = model.graph_layers[-1].attention_weights.mean(
      axis=1).squeeze()

  full_edges = zip(g.edges()[0], g.edges()[1])
  full_edges = [(u, v) for u, v in full_edges]
  assert len(full_edges) > 0
  for edge_id, edge in enumerate(full_edges):
    u = edge[0].item()
    v = edge[1].item()
    if u in nodes or v in nodes:
      if attention_weights[edge_id] > lower_bound:
        edges_to_plot.append((u, v))
        edges_ids.append(edge_id)
      edges_for_graphs.append((u, v))
      nodes_to_plot.add(u)
      nodes_to_plot.add(v)

  edges_to_plot = sorted(edges_to_plot)
  edges_for_graphs = sorted(edges_for_graphs)

  nodes_to_plot = np.array(sorted(list(nodes_to_plot)))
  nodes_labels = labels.cpu().numpy()[nodes_to_plot]
  edges_ids = np.array(edges_ids)

  assert len(nodes_labels) == len(nodes_to_plot)
  assert len(edges_to_plot) > 0
  sample_att_weights = attention_weights.cpu().numpy()

  # create new small graph based on the reduced set of nodes
  graph = nx.Graph(edges_for_graphs)
  to_keep_nodes = list(max(nx.connected_components(graph), key=len))
  to_remove_nodes = [
      node for node in nodes_to_plot if node not in to_keep_nodes
  ]
  graph.remove_nodes_from(to_remove_nodes)

  edges_ids = to_args(full_edges, list(graph.edges))

  edges_and_ids = [
      (e, e_id)
      for e, e_id, att in zip(list(graph.edges), edges_ids, attention_weights)
      if att > lower_bound
  ]
  edges_to_plot = [e for e, _ in edges_and_ids]
  edges_ids = np.array([e_id for _, e_id in edges_and_ids])

  nodes_to_plot = to_keep_nodes
  nodes_labels = labels.cpu().numpy()[nodes_to_plot]

  reduced_att = sample_att_weights[edges_ids]
  assert len(reduced_att) == len(edges_to_plot)
  visualize_attentions(
      graph,
      reduced_att,
      ax=ax,
      nodes_labels=nodes_labels,
      nodes_to_plot=nodes_to_plot,
      edges_to_plot=edges_to_plot)
  ax.set_axis_off()
  sm = plt.cm.ScalarMappable(
      cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
  sm.set_array([])
  plt.colorbar(sm, fraction=0.046, pad=0.01)
  dir_folder = 'vis_att/all_classes/{}/{}/{}'.format(
      n_nodes_to_plot,
      args.data,
      args.samples_per_class,
  )
  os.makedirs(dir_folder, exist_ok=True)
  fig.savefig('{}/{}.png'.format(dir_folder, args.model), dpi=100)


def visualize_tree(
    graph: nx.Graph,
    attentions: np.array,
    nodes_labels: torch.Tensor,
    degree=None,
    model_name='',
    node_number=0,
    data='',
    max_att_layer=0,
    samples_per_class=0,
    best_epoch=None,
    reachable_edges=None,
    reachable_att=None,
    use_tree_position=True,
):
  """Visualize tree-based explanations for nodes.

  Colors each labels accordingly to their class.
  The edges thickness corresponds to the edge-weight
  Cuts off subtrees if one of the branch < lower_bound

  Args:
      graph (nx.Graph): graph to be plotted.
      attentions (np.array): attention weights.
      nodes_labels (torch.Tensor): labels of all nodes.
      degree (None, optional): degree of each node to reflect importance.
      model_name (str, optional): Which model was used.
      node_number (int, optional): id of the nodes.
      data (str, optional): results from the analysis.
      max_att_layer (int, optional): How many attention layers were used
      samples_per_class (int, optional): How many sample/class used.
      best_epoch (None, optional): which epoch was the best.
      reachable_edges (None, optional): Sparsified edges to plot.
      reachable_att (None, optional): Strength of sparsified edges.
      use_tree_position (bool, optional): Convert to tree or not?.
  """

  # write dot file to use with graphviz
  fig, ax = plt.subplots()
  fig, ax = plt.subplots(figsize=(19.20, 10.80))
  node_pos = graphviz_layout(graph, prog='dot')

  if reachable_att is not None and len(reachable_att) < len(attentions):
    attentions = np.array(reachable_att)
    edges_to_plot = reachable_edges
    # filter nodes as well.
    # flatten the list of edges to know which nodes to plot
    nodes_to_plot = set(sum(reachable_edges, ()))

    if use_tree_position:
      graph = nx.DiGraph(edges_to_plot)
      node_pos = graphviz_layout(graph, prog='dot')
    else:
      node_pos = graphviz_layout(graph, prog='dot')
      node_pos = {node: node_pos[node] for node in nodes_to_plot}
  else:
    edges_to_plot = graph.edges()
    attentions = np.array(attentions)
    nodes_to_plot = graph.nodes()
    node_pos = graphviz_layout(graph, prog='dot')


  assert len(attentions) == len(edges_to_plot),\
         'not matching length %d % d' % (len(attentions), len(edges_to_plot))
  nx.draw_networkx_edges(
      graph,
      node_pos,
      edgelist=edges_to_plot,
      edge_color=attentions,
      edge_cmap=plt.cm.Reds,
      width=attentions * 10,
      alpha=0.5,
      ax=ax,
      edge_vmin=0,
      edge_vmax=1,
  )
  nodes_colors = sns.color_palette('deep', max(nodes_labels) + 1)
  colors = [nodes_colors[nodes_labels[v]] for v in nodes_to_plot]
  if degree is None:
    degree = nx.degree(graph)
  node_size = [degree[v] * 10 for v in nodes_to_plot]

  nx.draw_networkx_nodes(
      graph,
      node_pos,
      nodelist=nodes_to_plot,
      ax=ax,
      node_color=colors,
      alpha=0.9,
      node_size=node_size,
  )
  ax.set_axis_off()
  sm = plt.cm.ScalarMappable(
      cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
  sm.set_array([])
  plt.colorbar(sm, fraction=0.046, pad=0.01)
  # same layout using matplotlib with no labels
  plt.title('%s Explanation of node %d' % (model_name, node_number))
  folder = 'explain_vis/%s/%s/%s/' % (data, samples_per_class, max_att_layer)
  os.makedirs(folder, exist_ok=True)
  file_name = '{2}/{1}_{0}_{3}.png'.format(model_name, node_number, folder,
                                           best_epoch)

  print('Saving images to %s' % file_name)
  fig.savefig(file_name, dpi=100)


def reverse_edge(edges):
  """Reverse the edges list to plot in tree format

  This is required, otherwise the tree will be flipped during plotting.

  Args:
      edges (list): convert edges from src->trg to trg->src. Required to plot.

  Returns:
      list: reverted_egdes
  """
  reverted_egdes = [(v, u) for u, v in edges]
  return reverted_egdes


def collect_attentions(graph, attention_list, node):
  """Collect the right edges and nodes needed to visualize the graph.

  Avoid looping.

  Args:
      graph (DGLGraph): the entire graph.
      attention_list: list of multiple layer. Each is a vector. The first
        attention is for the first layer
      node: id required to plot. Return

  Returns:
      reduced_graph: the smaller graph going out from the node only.
      all_att_weights: the attention weights needed for plotting.
      reachable_edges: reduced list of edges, involved in the new graph.
      reachable_att, degree: The Strength of these edges.
  """
  edges = graph.edges()
  edges_to_id = {(src.item(), trg.item()): edge_id
                 for edge_id, (src, trg) in enumerate(zip(edges[0], edges[1]))}
  if isinstance(graph, dgl.DGLGraph):
    graph = graph.cpu().to_networkx()

  neighbors_list = graph.neighbors(node)
  neighbors_pairs_list = [(neighbor, node) for neighbor in neighbors_list]
  all_att_weights = []
  reachable_att: list = []
  final_edges: list = []
  reachable_edges: list = []
  collected_nodes: set = {node}
  reachable_nodes: set = {node}

  attention_weights_from_orig = [
      attention_list[0][edges_to_id[pair]] for pair in neighbors_pairs_list
  ]
  sum_att = np.sum(attention_weights_from_orig)
  assert sum_att >= .99, 'Why are some attention_weights missing? %f' % sum_att

  for att_weights_layer in attention_list:
    next_hops_neighbors = []
    for neighbors_pair in neighbors_pairs_list:
      if neighbors_pair not in final_edges and \
         (neighbors_pair[1], neighbors_pair[0]) not in final_edges:
        edge_id = edges_to_id[neighbors_pair]
        att_weight = att_weights_layer[edge_id].item()
        all_att_weights.append(att_weight)
        final_edges.append(neighbors_pair)

        # Add more neighbors to the list
        trg = neighbors_pair[0]
        src = neighbors_pair[1]
        direct_neighbors = graph.neighbors(trg)
        direct_neighbors_pairs = [(neighbor, trg)
                                  for neighbor in direct_neighbors
                                  if neighbor not in collected_nodes]

        collected_nodes.add(trg)
        # In addition, collect a reduced edge list for plotting
        if src in reachable_nodes and att_weight > 0:
          reachable_nodes.add(trg)
          reachable_edges.append(neighbors_pair)
          reachable_att.append(att_weight)
        next_hops_neighbors.extend(direct_neighbors_pairs)
    neighbors_pairs_list = next_hops_neighbors
  assert len(all_att_weights) == len(final_edges)
  # needs to revert edges to plot in a tree format
  reduced_graph = nx.DiGraph(reverse_edge(final_edges))

  assert len(all_att_weights) == len(final_edges)
  assert len(all_att_weights) == len(reduced_graph.edges)
  assert len(reachable_edges) <= len(final_edges)
  assert len(reachable_edges) == len(reachable_att)
  # needs to revert to support tree format
  reachable_edges = reverse_edge(reachable_edges)
  degree = nx.degree(graph)
  return reduced_graph, all_att_weights, reachable_edges, reachable_att, degree


def plot_att_trees(
    graph,
    attention_list,
    nodes_labels,
    model_name='',
    data='',
    samples_per_class=None,
    best_epoch=None,
    nodes=None,
):
  """Chooses some nodes to plot.

  Start with random nodes for plotting.
  Afterwards, choose some nodes for plotting with SparseMax.

  Args:
      graph (DGLGraph): the entire graph.
      attention_list: attentinos from all layers.
      nodes_labels (TYPE): labels of all nodes.
      model_name (str, optional): which model was used.
      data (str, optional): Results of the analysis.
      samples_per_class (None, optional): how many samples/class used.
      best_epoch (None, optional): which was the best epoch.
      nodes (None, optional): which nodes to plot.
  """
  assert len(attention_list) >= 1, 'Missing attentions?'
  assert attention_list[0].ndim == 1, 'Attentions heads are not squeezed'

  if nodes is None:
    nodes_idx = [1701, 729, 2594, 1538, 577, 643, 1899, 2196]
    nodes = [graph.nodes()[idx] for idx in nodes_idx]

  reversed_attention_list = attention_list[::-1]
  for max_att_layer in range(2, len(reversed_attention_list) + 1):
    # for max_att_layer in range(2, len(reversed_attention_list)):
    attentions = reversed_attention_list[:max_att_layer]
    for node in nodes:
      node = node.item()
      results = collect_attentions(
          graph,
          attentions,
          node,
      )
      reduced_graph, attention, reachable_edges, reachable_att, degree = results
      assert nodes_labels is not None
      visualize_tree(
          reduced_graph,
          attention,
          nodes_labels,
          degree=degree,
          model_name=model_name,
          node_number=node,
          data=data,
          max_att_layer=max_att_layer,
          samples_per_class=samples_per_class,
          best_epoch=best_epoch,
          reachable_edges=reachable_edges,
          reachable_att=reachable_att,
      )
  print('Visualization successful')


if __name__ == '__main__':
  graph = nx.DiGraph()

  graph.add_node('ROOT')

  for i in range(5):
    graph.add_node('Child_%i' % i)
    graph.add_node('Grandchild_%i' % i)
    graph.add_node('Greatgrandchild_%i' % i)

    graph.add_edge('ROOT', 'Child_%i' % i)
    graph.add_edge('Child_%i' % i, 'Grandchild_%i' % i)
    graph.add_edge('Grandchild_%i' % i, 'Greatgrandchild_%i' % i)
  nodes_labels = np.random.randint(5, size=(16,))
  attention = np.random.random(size=(15,))
  visualize_tree(graph, attention, nodes_labels)
  print('Visualization successful')
