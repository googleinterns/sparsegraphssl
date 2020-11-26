"""Tests the graph analysis module."""

import dgl
import numpy as np
import torch
from analysis.attention_analysis import measure_homophily
from analysis.attention_analysis import measure_entropy
from analysis.attention_analysis import measure_ref_uniform_entropy
from analysis.attention_analysis import plot_attention_entropy
from analysis.attention_analysis import plot_unlabeled_samples
from analysis.attention_analysis import compute_att_on_shortest_paths
from analysis.attention_vis import plot_att_trees

from data_reader.data_utils import sample_heterophil_edges
import pytest
from dgl.nn.pytorch.softmax import edge_softmax


@pytest.fixture
def karate_club_graph():
  """Constructs the karate_club_graph.

  Returns:
      graph: karate_club_graph
  """
  # All 78 edges are stored in two numpy arrays. One for source endpoints
  # while the other for destination endpoints.
  src = np.array([
      1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10, 10, 11, 12,
      12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21, 25, 25, 27, 27, 27,
      28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33
  ])
  dst = np.array([
      0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4, 5, 0, 0, 3, 0,
      1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23, 24, 2, 23, 26, 1, 8, 0,
      24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 8, 9, 13, 14, 15,
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32
  ])
  # Edges are directional in DGL; Make them bi-directional.
  u = np.concatenate([src, dst])
  v = np.concatenate([dst, src])
  graph = dgl.DGLGraph((u, v))
  n_nodes = len(graph.nodes())
  n_edges = len(graph.edges()[0])
  labels = torch.randint(2, (n_nodes,))
  # labels[:16] = 0
  edges_vector = torch.rand((n_edges,))
  attention_weights = edge_softmax(graph, edges_vector)
  assert len(attention_weights) == n_edges

  return graph, attention_weights, labels


def test_homophily(karate_club_graph):
  """Tests homomphily.

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph

  mean_homophily = measure_homophily(graph, attention_weights, labels)
  print('homophily: {}'.format(mean_homophily))
  assert mean_homophily > 0,\
         'homophily is negative: {}'.format(mean_homophily)
  assert mean_homophily < 1,\
         'homophily is larger than 1: {}'.format(mean_homophily)


def test_entropy_measurement(karate_club_graph):
  """Test entropy measurement accross different nodes.

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph
  nodes_entropy, mean_entropy = measure_entropy(graph, attention_weights)
  print(nodes_entropy[:10])
  print('Custom Entropy {}'.format(mean_entropy))
  assert (nodes_entropy >= 0).all(),\
         'Entropy cannot be negative {}'.format(nodes_entropy)


def test_uniform_entropy_measurement(karate_club_graph):
  """Tests uniform_entropy measurement.

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph
  nodes_entropy, mean_entropy = measure_ref_uniform_entropy(
      graph, attention_weights.shape)
  print(nodes_entropy[:10])
  print('Random entropy {}'.format(mean_entropy))
  assert (nodes_entropy >= 0).all(),\
         'Entropy cannot be negative {}'.format(nodes_entropy)


def test_plot_entropy(karate_club_graph):
  """Test plotting entropy

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph
  n_heads = 4
  attention_weights_n_heads = torch.stack([
      attention_weights,
  ] * n_heads, -1)
  plot_attention_entropy(
      graph,
      attention_weights_n_heads,
      data='cora',
      samples_per_class=1,
      model_name='sparsegat',
  )


@pytest.mark.plot_train
def test_plot_unlabeled_samples(karate_club_graph):
  """Test plootting unlabeled samples.

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph
  train_mask = torch.zeros(labels.shape).bool()
  train_mask[:10] = True
  plot_unlabeled_samples(
      graph=graph,
      attention_weights=attention_weights,
      labels=labels,
      train_mask=train_mask,
      dataset='karate_club_graph',
      model_name='Random',
      samples_per_class=1,
  )


@pytest.mark.homophil_edges
def test_homophil_edges(karate_club_graph):
  """Test attention on homophily edges.

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph
  noisy_edges_src, noisy_edges_trg = sample_heterophil_edges(labels, 10)
  print(noisy_edges_src, noisy_edges_trg)


@pytest.mark.shortest_path
def test_shortest_path(karate_club_graph):
  """Test attention on shortest_paths.

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph
  # labels[:10] = 1
  att_mean = compute_att_on_shortest_paths(graph, attention_weights, labels)
  assert att_mean > 0, 'Att_mean %f should be positive' % att_mean
  print('Att_mean on shortest paths %f' % att_mean)
  assert att_mean < .1, 'Att_mean should be very small %d' % att_mean


@pytest.mark.tree
def test_tree(karate_club_graph):
  """Tests interpretability tree.

  Args:
      karate_club_graph (nx.Graph): the karate_club_graph.
  """
  graph, attention_weights, labels = karate_club_graph
  att_list = [
      attention_weights,
  ] * 3

  plot_att_trees(graph, att_list, labels, model_name='Test')
