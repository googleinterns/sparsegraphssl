"""Utilizations to process and load data."""

import os

import torch
import numpy as np
import random
import scipy.io as sio
import networkx as nx
import pickle
from dgl import DGLGraph
import json


def sample_subgraphs_from_ppi(n_subgraphs_requested, seed=None):
  """ Pick from 20 subgraphs randomly some subgraphs.

  If n_subgraphs_requested = -1 use labeling_rate only
  else use n_subgraphs_requested directly
  return
      train_idx_random_selected

  Args:
      n_subgraphs_requested (TYPE): Description
      seed (None, optional): Description

  Returns:
      list: randomly selected nodes for training.
  """
  all_train_idx = np.arange(1, 21)
  set_seed(seed)
  random_train_idx = np.random.permutation(all_train_idx)
  train_idx_random_selected = random_train_idx[:n_subgraphs_requested]
  return train_idx_random_selected


def set_seed(seed=None):
  """Set all seeds for deterministic computation.

  Args:
      seed (None, optional): which number to be used for seeding.
  """
  if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_subsampled_train_idx(graph_id, n_subgraphs_used, seed=None):
  """Subamples points from ppi-graph.

  Takes out one graph, and subsample from there.

  Args:
      graph_id (int): Which subgraph of ppi to be used.
      n_subgraphs_used (int): how many subgraphs to be used.
      seed (None, optional): which random seed should be used.

  Returns:
      list: node indices for training.
  """
  train_indices_all = np.arange(1, 21)
  set_seed(seed)
  random_train_subgraph_idx = np.random.permutation(train_indices_all)
  selected_subgraphs = random_train_subgraph_idx[:n_subgraphs_used]
  # pdb.set_trace()
  assert len(selected_subgraphs) == n_subgraphs_used
  train_indices = []
  for train_graph_id in selected_subgraphs:
    train_graph_indx = np.where(graph_id == train_graph_id)[0]
    train_indices.extend(train_graph_indx)
  # pdb.set_trace()
  assert len(train_indices) >= 0
  return train_indices


def get_all_traing_indices(graph_id):
  """Summary

  Args:
      graph_id (TYPE): Description

  Returns:
      list: node indices for training.
  """
  train_indices = []
  for train_graph_id in np.arange(1, 21):
    train_graph_indx = np.where(graph_id == train_graph_id)[0]
    train_indices.extend(train_graph_indx)
  assert len(train_indices) >= 0
  return train_indices


def subsample_ppi(n_subgraphs_used, labeling_rate, graph_id, seed=None):
  """Summary

  Args:
      n_subgraphs_used (TYPE): Description
      labeling_rate (TYPE): Description
      graph_id (TYPE): Description
      seed (None, optional): Description

  Returns:
      list: node indices for training.
  """
  assert seed is not None
  all_train_idx = get_all_traing_indices(graph_id)

  subsampled_train_idx = get_subsampled_train_idx(
      graph_id, n_subgraphs_used, seed=seed)
  n_train_samples = len(all_train_idx)
  requested_samples = int(n_train_samples * labeling_rate)
  assert requested_samples <= len(
      subsampled_train_idx), 'requested {} from {} samples'.format(
          requested_samples, len(subsampled_train_idx))
  set_seed(seed)
  random_train_idx = np.random.permutation(subsampled_train_idx)
  train_idx_random_selected = random_train_idx[:requested_samples]
  # pdb.set_trace()
  return train_idx_random_selected


def get_full_train_mask(train_mask, val_mask, test_mask, n_samples):
  """Summary

  Args:
      train_mask (TYPE): Description
      val_mask (TYPE): Description
      test_mask (TYPE): Description
      n_samples (TYPE): Description

  Returns:
      list: node indices for training.
  """
  # get all indices
  val_idx = val_mask.nonzero()[0]
  test_idx = test_mask.nonzero()[0]
  all_samples_indx = set(range(0, n_samples))
  full_train_indices = all_samples_indx
  full_train_indices = full_train_indices - set(val_idx)
  full_train_indices = full_train_indices - set(test_idx)
  full_train_indices = list(full_train_indices)
  return full_train_indices


def sample_train_mask(train_idx, args, labels, seed=None):
  """Summary

  Args:
      train_idx (TYPE): Description
      args (TYPE): Description
      labels (TYPE): Description
      seed (None, optional): Description

  Returns:
      list: node indices for training.
  """
  samples_per_class = args.samples_per_class

  if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
  samples_per_class = args.samples_per_class
  train_idx_random_selected = sample_fixed_number_per_class(
      train_idx, labels, samples_per_class)
  print(train_idx_random_selected)
  return train_idx_random_selected


def sample_imbalanced_binary_train_mask(train_idx, args, labels, seed=None):
  """For strongly imbalance binary masks, we need to be more accurate.

  Changing the number of samples from the minority class otherwise would
  have a too large effect.
  For sampling using labeling_rates: #anomaly_samples  =
  #total_number_anomaly_samples * labeling_ratio.

  @return:

  Args:
      train_idx (TYPE): Description
      args (TYPE): Description
      labels (TYPE): Description
      seed (None, optional): Description

  Returns:
      list: node indices for training.
  """
  samples_per_class = args.samples_per_class
  if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
  if samples_per_class != -1:
    samples_per_class = args.samples_per_class
    train_idx_random_selected = sample_fixed_number_per_class(
        train_idx, labels, samples_per_class)
  else:
    n_samples_anomaly_total = count_samples_in_class(
        train_idx, labels, class_id=1)
    n_samples_normal_total = count_samples_in_class(
        train_idx, labels, class_id=0)
    n_anomaly_samples = int(n_samples_anomaly_total * args.labeling_rate)
    n_normal_samples = int(n_anomaly_samples * n_samples_anomaly_total /
                           n_samples_normal_total)
    custom_samples_per_class_list = [n_normal_samples, n_anomaly_samples]
    train_idx_random_selected = sample_fixed_number_per_class(
        train_idx,
        labels,
        samples_per_class=None,
        custom_samples_per_class_list=custom_samples_per_class_list)
  return train_idx_random_selected


def count_samples_in_class(train_idx, labels, class_id=None):
  """ Counts how many samples are in each class

  This fucntions performs some testing for correctness.

  @args:
      train_idx: only these indices are used for training.
      labels: all labels. Might contain valid and test data as well
      class:_id numerical value of the class id
  @return:
      n_samples_in_class: how many samples are present in the training data of
      each class

  Args:
      train_idx (list): which nodes should be used for training.
      labels (torch.LongTensor): labels for all nodes.
      class_id (None, optional): the class to be tested

  Returns:
      list: node indices for training.
  """
  assert class_id is not None, 'missing argument'
  n_samples_in_class = (labels[train_idx] == class_id).sum()
  assert n_samples_in_class > 0, 'something went wrong'
  assert n_samples_in_class < len(
      labels), 'data is too homogenous. Data spliting might be wrong.'
  return n_samples_in_class


def sample_fixed_number_per_class(train_idx,
                                  labels,
                                  samples_per_class,
                                  custom_samples_per_class_list=None):
  """Takes some samples per class.

  Constructs a dataset by sampling given a fixed number/class.

  Args:
      train_idx (list): Which samples are in the traninig set.
      labels (torch.LongTensor): Labels of all nodes
      samples_per_class (int): How mnay samples/class to be usde.
      custom_samples_per_class_list (None, optional): Changeable/class.

  Returns:
      results: combined_train_indices with n samples/class
  """
  train_idx = np.array(train_idx)
  all_class = range(0, labels.max() + 1)
  train_labels = labels[train_idx]
  train_idx_random_selected = []
  n_classes = train_labels.max() + 1
  if custom_samples_per_class_list is None:
    samples_per_class_list = [samples_per_class] * n_classes
  else:
    samples_per_class_list = custom_samples_per_class_list

  assert len(
      samples_per_class_list
  ) == n_classes, 'missing requested samples for some classes {}'.format(
      samples_per_class_list)

  for class_number, n_samples_this_class in zip(all_class,
                                                samples_per_class_list):
    train_samples_from_this_class = train_idx[train_labels == class_number]
    random_ordered_samples = np.random.permutation(
        train_samples_from_this_class)
    sampled_train_idx = random_ordered_samples[:n_samples_this_class]

    assert len(
        sampled_train_idx
    ) == n_samples_this_class, 'Requesting too many samples {} from max {}'.format(
        n_samples_this_class, len(sampled_train_idx))
    train_idx_random_selected.append(sampled_train_idx)

  results = np.concatenate(train_idx_random_selected, axis=0)
  return results


def remove_indices(idx, to_remove_idx):
  """Removes some nodes from a list.

  Args:
      idx (list): full-list of nodes.
      to_remove_idx (list): list of nodes to be removed.

  Returns:
      list: node indices for training.
  """
  remaining_idx = np.array(list(set(list(idx)) - set(list(to_remove_idx))))
  return remaining_idx


def sample(idx, seed, num):
  """Summary

  Args:
      idx (list): list of all node idx.
      seed (int): random seed.
      num (int): how many samples to use for training.

  Returns:
      inx_random: list of random nodes.
  """
  if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
  random_train_idx = np.random.permutation(idx)
  idx_random = random_train_idx[:num]
  return idx_random


def save_pickle(cache_file, data):
  """Save pickle files.

  Args:
      cache_file (str): where to save the file.
      data (json): resutls to be saved
  """
  with open(cache_file, 'wb') as f:
    pickle.dump(data, f)


def read_pickle(cache_file):
  """Load pickle files

  Args:
      cache_file (str): file_name to read from.

  Returns:
      data: read data.
  """
  with open(cache_file, 'rb') as f:
    data = pickle.load(f)
  return data


def append_pickle(cache_file, data):
  """Append to files.

  Args:
      cache_file (str): file_name to append to.
      data (json): Content to append to the files.
  """
  with open(cache_file, 'a') as f:
    json.dump(data, f)
    f.write('\n')


def exec_if_not_done(file='', func=None, redo=False):
  """Decorator to save computation time.

  Args:
      file (str, optional): which file is involed.
      func (None, optional): which fucntion should be redone.
      redo (bool, optional): force redo or not.

  Returns:
      results: all results
  """
  if not os.path.exists(file) or redo:
    os.makedirs(os.path.dirname(file), exist_ok=True)
    results = func()
    save_pickle(file, results)
  else:
    results = read_pickle(file)
  return results


def get_labeling_mask(list_of_masks):
  """Constructs a labeling mask.

  Based on the list of masks, construct labeling masks.
  first mask = train
  unlabeled = 0
  labeled = 1
  valid and test = 1

  Args:
      list_of_masks (TYPE): Description

  Returns:
      Torch vector to mark which sample is unlabeled
  """
  assert len(list_of_masks) == 3
  combined_mask = list_of_masks[0] + list_of_masks[1] + list_of_masks[2]
  reweighting_vector = torch.LongTensor(combined_mask.astype('double'))

  return reweighting_vector


def create_graph(g):
  """Creates a graph with self-loops.

  Args:
      g (g): graph

  Returns:
      TYPE: Description
  """
  g.remove_edges_from(nx.selfloop_edges(g))
  g = DGLGraph(g)
  g.add_edges(g.nodes(), g.nodes())
  return g


def zip_edge(edge_double_list):
  """Zips edges into a single list.

  Args:
      edge_double_list (tuple): (src, trg). Zips into  sinlge list of tuple
        (src, trg)

  Returns:
      zipped_edges: single list of edges.
  """
  edge_zipped = [(u.item(), v.item()) for u, v in zip(*edge_double_list)]
  return edge_zipped


def sample_heterophil_edges(labels, n_edges_to_add):
  """Sample edges coming from different classes.

  Sample a bit more than required to remove samples comming from the sample
  class.

  Args:
      labels: torch.Tensor: all labels starting from 0->max
      n_edges_to_add: how many edges to add to the graph

  Returns:
      noisy_edges_src: src of noisy edges.
      noisy_edges_trg: trg of noisy edges.
  """

  if type(labels) == torch.Tensor:
    labels = labels.data.cpu().numpy()
  max_class = labels.max() + 1
  sample_tensor = torch.zeros((n_edges_to_add, max_class)).long()
  over_sampling = n_edges_to_add * 4
  for cl in range(max_class):
    samples_this_class = (labels == cl).nonzero()[0].squeeze()
    # sample from certain class
    random_choice = np.random.choice(samples_this_class, n_edges_to_add)
    sample_tensor[:, cl] = torch.Tensor(random_choice)
  # sample without repetition
  sample_pairs = np.random.choice(max_class, size=(over_sampling, 2))
  # delete samples which comes from the same class
  hetero_pairs = sample_pairs[sample_pairs[:, 0] != sample_pairs[:, 1]]
  assert hetero_pairs.shape[0] > n_edges_to_add, \
         'Not enough pairs %d' % hetero_pairs.shape[0]
  final_pairs = hetero_pairs[:n_edges_to_add]

  noisy_edges_src = sample_tensor[range(n_edges_to_add), final_pairs[:, 0]]
  noisy_edges_trg = sample_tensor[range(n_edges_to_add), final_pairs[:, 1]]
  return noisy_edges_src, noisy_edges_trg


def add_noise_to_graph(graph, noise_ratio=1, noise_type='random', labels=None):
  """Adds noisy connections to graph.

  Add random edges between random edges.

  Args:
      graph (TYPE): Description
      noise_ratio: relative to current number of edges.
      noise_type: Ways to generate nois: random, homophily, n-hops, shortest
        paths Returns
      labels (None, optional): Description

  Returns:
      noisy_graph: a graph with noisy edges.
      noisy_edges_ids: ids of the noisy edges.

  Raises:
      NotImplementedError: Some types are not implemented yet.
      ValueError: Other noise_types are not supported.
  """
  graph = graph.local_var()
  nodes = graph.nodes()
  edges = set(zip_edge(graph.edges()))
  n_edges = len(edges)
  n_nodes = len(nodes)
  n_edges_to_add = int(n_edges * noise_ratio)
  if noise_ratio == 0:
    noisy_edges_src = nodes[torch.randint(n_nodes, (0,))]
    noisy_edges_trg = nodes[torch.randint(n_nodes, (0,))]
  else:
    if noise_type == 'random':
      # transforms clasess to their classes.
      noisy_edges_src = nodes[torch.randint(n_nodes, (n_edges_to_add,))]
      noisy_edges_trg = nodes[torch.randint(n_nodes, (n_edges_to_add,))]
    elif noise_type == 'heterophily':
      assert labels is not None
      noisy_edges_src, noisy_edges_trg = sample_heterophil_edges(
          labels, n_edges_to_add)
      noisy_edges_src = noisy_edges_src.type_as(nodes)
      noisy_edges_trg = noisy_edges_trg.type_as(nodes)

    elif noise_type == 'anti_n_hops':
      raise NotImplementedError
    elif noise_type == 'anti_shortest_paths':
      raise NotImplementedError
    else:
      raise ValueError('Unknown noise type %s ' % noise_type)

  graph.add_edges(noisy_edges_src, noisy_edges_trg)
  noisy_graph = graph
  noisy_edges_ids = list(range(n_edges, n_edges + n_edges_to_add))
  assert graph.edges()[0].shape[0] == n_edges + n_edges_to_add
  return noisy_graph, noisy_edges_ids
