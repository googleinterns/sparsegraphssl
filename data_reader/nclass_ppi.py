"""PPI Dataset."""
from data_reader.legacy_ppi import PPIDataset
import numpy as np
from data_reader.data_utils import sample_subgraphs_from_ppi
from data_reader.data_utils import subsample_ppi

_url = 'dataset/ppi.zip'
import torch as th


class LowLabelPPIDataset(PPIDataset):
  """Low-labeling ppi dataset

  Attributes
  ----------
  args : Namespace.
      Arguments to configure ppi-dataset
  subsampling : str
      Whether to use subsampling or not.
  test_graphs : list
      list of sub-graphs for testing.
  test_labels : list
      list of labels for testing.
  test_mask_list : list
      list of masks for testing.
  train_graphs : list
      list of sub-graphs for training.
  train_labels : list
      list of labels for training.
  train_mask_list : list
      list of masks for testing.
  valid_graphs : list
      list of sub-graphs for validation.
  valid_labels : list
      list of labels for validation.
  valid_mask_list : list
      list of masks for validation.
  """

  def __init__(self, mode, args, subsampling=False):
    """Initialize the dataset.

    This functions inits the ppi-dataset.


    Args
    ----------
    mode : str
        ('train', 'valid', 'test').
    args : Namespace.
        Arguments to configure ppi-dataset.
    subsampling : bool, optional
        Use subsampling or not.
    """
    self.args = args
    self.subsampling = subsampling
    super(LowLabelPPIDataset, self).__init__(mode)

  def _preprocess(self):
    """Summary"""
    if self.mode == 'train':
      n_subgraphs_requested = self.args.samples_per_class
      train_subgraphs_idx = sample_subgraphs_from_ppi(
          n_subgraphs_requested, seed=self.args.repeated_runs)
      self.train_mask_list = []
      self.train_graphs = []
      self.train_labels = []
      for train_graph_id in train_subgraphs_idx:
        train_graph_mask = np.where(self.graph_id == train_graph_id)[0]
        self.train_mask_list.append(train_graph_mask)
        subgraph = self.graph.subgraph(train_graph_mask)
        graph = subgraph
        graph.readonly(False)
        self.train_graphs.append(graph)
        self.train_labels.append(self.labels[train_graph_mask])

    if self.mode == 'valid':
      self.valid_mask_list = []
      self.valid_graphs = []
      self.valid_labels = []
      for valid_graph_id in range(21, 23):
        valid_graph_mask = np.where(self.graph_id == valid_graph_id)[0]
        self.valid_mask_list.append(valid_graph_mask)

        subgraph = self.graph.subgraph(valid_graph_mask)
        graph = subgraph
        graph.readonly(False)

        self.valid_graphs.append(graph)
        self.valid_labels.append(self.labels[valid_graph_mask])
    if self.mode == 'test':
      self.test_mask_list = []
      self.test_graphs = []
      self.test_labels = []
      for test_graph_id in range(23, 25):
        test_graph_mask = np.where(self.graph_id == test_graph_id)[0]
        self.test_mask_list.append(test_graph_mask)
        subgraph = self.graph.subgraph(test_graph_mask)
        graph = subgraph
        graph.readonly(False)

        self.test_graphs.append(graph)
        self.test_labels.append(self.labels[test_graph_mask])


class CustomPPIDataset(LowLabelPPIDataset):
  """A simple, standard ppi-dataset."""

  def __getitem__(self, item):
    """Get the i^th sample.

    Args
    ----------
      item : int
          The sample index.

    Returns
    -------
      (dgl.DGLGraph, ndarray, ndarray)
          The graph, features and its label.

    """
    if self.mode == 'train':

      return self.train_graphs[item], self.features[
          self.train_mask_list[item]], self.train_labels[item]
    if self.mode == 'valid':
      return self.valid_graphs[item], self.features[
          self.valid_mask_list[item]], self.valid_labels[item]
    if self.mode == 'test':
      return self.test_graphs[item], self.features[
          self.test_mask_list[item]], self.test_labels[item]


def compute_unsup_weights(train_indices, labeled_items):
  """Transforms global indices to local indices.

  Args
  ----------
    train_indices : list
        list of training indices.
    labeled_items : list
        which samples are labeled.

  Returns
  -------
    torch.BoolTensor
        Weights to mark which samples are labeled and which not.
  """
  unsup_sparsemax_weights = np.zeros((len(train_indices),))
  for indx in labeled_items:
    transformed_idx = np.where(train_indices == indx)
    unsup_sparsemax_weights[transformed_idx] = 1
  assert unsup_sparsemax_weights.sum() <= len(
      train_indices), 'setting weights not successful'
  unsup_sparsemax_weights = th.FloatTensor(unsup_sparsemax_weights)
  return unsup_sparsemax_weights


class SubsampledPPIDataset(LowLabelPPIDataset):
  """Sample entire subgraphs for use.

  Attributes
  ----------
  graph_id : int
      which subgraph to subsample from.
  train_graphs : list
      graphs used for training.
  train_labels : list
      labels of training dataset.
  train_mask_list : list
      binary mask to mark training samples.
  unsup_sparsemax_weights : list
      weights of labeled/unlabeled samples.
  """

  def _preprocess(self):
    """Preprocess the dataset."""

    self.unsup_sparsemax_weights = []
    if self.mode == 'train':
      n_subgraphs_used = self.args.samples_per_class
      total_train_graph_indices = subsample_ppi(
          n_subgraphs_used,
          self.args.labeling_rate,
          self.graph_id,
          seed=self.args.repeated_runs)

      self.train_mask_list = []
      self.train_graphs = []
      self.train_labels = []
      for train_graph_id in range(1, 21):
        train_graph_indices_this_id = np.where(
            self.graph_id == train_graph_id)[0]
        labeled_items_this_id = np.array(
            list(
                set(total_train_graph_indices)
                & set(train_graph_indices_this_id)))
        # pdb.set_trace()
        print('Training samples in graph_id {} {}/{}'.format(
            train_graph_id, len(labeled_items_this_id),
            len(train_graph_indices_this_id)))
        if len(labeled_items_this_id) == 0:
          # this subgraph contains no sample for training.
          # very unlikely, but could indeed happen.
          continue
        # unsupervised_mode needs a weights vector additionally
        unsup_sparsemax_weights = compute_unsup_weights(
            train_graph_indices_this_id, labeled_items_this_id)
        self.unsup_sparsemax_weights.append(unsup_sparsemax_weights)
        train_graph_indices = train_graph_indices_this_id

        self.train_mask_list.append(train_graph_indices)
        subgraph = self.graph.subgraph(train_graph_indices)
        graph = subgraph
        graph.readonly(False)
        self.train_graphs.append(graph)
        self.train_labels.append(self.labels[train_graph_indices])

    if self.mode in ['valid', 'test']:
      super()._preprocess()

  def __getitem__(self, item):
    """Get the i^th sample.

    Paramters
    ---------
      item : int
          The sample index.

    Returns
    -------
      (dgl.DGLGraph, ndarray, ndarray)
          The graph, features and its label.

    """
    if self.mode == 'train':
      unsup_sparsemax_weights = self.unsup_sparsemax_weights[item]
      labeled_items = (unsup_sparsemax_weights).nonzero()
      assert max(labeled_items) <= len(unsup_sparsemax_weights)
      assert len(unsup_sparsemax_weights) == len(self.train_graphs[item])
      return self.train_graphs[item], self.features[
          self.train_mask_list[item]], self.train_labels[
              item], labeled_items, unsup_sparsemax_weights
    if self.mode == 'valid':
      return self.valid_graphs[item], self.features[
          self.valid_mask_list[item]], self.valid_labels[item]
    if self.mode == 'test':
      return self.test_graphs[item], self.features[
          self.test_mask_list[item]], self.test_labels[item]


class UnsupSubgraphPPIDataset(PPIDataset):
  """Sample from each subgraph some samples.

  Attributes
  ----------
  args : Namespace.
      Arguments to configure ppi-dataset
  is_labeled_list : list
      Mark if samples are labeled.
  labeled_weights : torch.Tensor
      Weights for labeled samples.
  subsampling : bool
      Whether to use subsampling or not.
  """

  def __init__(self, mode, args, subsampling=False):
    """Initialize the dataset.


    Args
    ----------
    mode : str
        ('train', 'valid', 'test').
    args : Namespace.
        Arguments to configure ppi-dataset
    subsampling : bool, optional
        Use subsampling or not
    """
    self.args = args
    self.subsampling = subsampling
    super(UnsupSubgraphPPIDataset, self).__init__(mode)

  def _preprocess(self):
    """ Preprocess dataset.

    provide all data as usual.(legacy ppi-dataset)
    postprocess:
        + set labels of non-chosen subgraphs to -1 (*0 -1)
        + provide additonal labeled mask (similar to unsupervised_weights)
            * zero for unlabeled.
            * ones for labeled
    """

    super()._preprocess()
    train_idx_labeled = sample_subgraphs_from_ppi(
        n_subgraphs_requested=self.args.samples_per_class,
        seed=self.args.repeated_runs)
    train_idx_unlabeled = list(set(range(1, 21)) - set(train_idx_labeled))
    assert len(train_idx_labeled) + len(
        train_idx_unlabeled) == 20, 'Missing subgraphs {} {}'.format(
            len(train_idx_labeled), len(train_idx_unlabeled))
    is_labeled_list = []
    labeled_weights = []
    for item in range(1, 21):
      """
            mask labels
            create is_labeled vector
            """
      shifted_item = item - 1
      labels = self.train_labels[shifted_item]
      n_samples = len(labels)
      if item in train_idx_unlabeled:
        # since the ids start at 1, the items will be shifted
        # print(shifted_item)
        unsupervised_labels = (labels * 0) - 1
        self.train_labels[shifted_item] = unsupervised_labels
        is_labeled = th.zeros((n_samples,))
      else:
        is_labeled = th.ones((n_samples,))
      assert is_labeled.shape[0] == n_samples, '{} {}'.format(
          is_labeled.shape[0], n_samples)
      is_labeled = is_labeled.bool()
      is_labeled_list.append(is_labeled)
      labeled_weights.append(is_labeled.float())
    self.is_labeled_list = is_labeled_list
    self.labeled_weights = labeled_weights
    assert len(is_labeled_list) == len(self.train_labels)

  def __getitem__(self, item):
    """Get the i^th sample.

    Args
    ---------
      item : int
          The sample index.

    Returns
    -------
      (dgl.DGLGraph, ndarray, ndarray)
          The graph, features and its label.

    """
    if self.mode == 'train':
      is_labeled = self.is_labeled_list[item]
      labeled_weights = self.labeled_weights[item]
      assert len(is_labeled) == len(
          self.train_labels[item]), 'not matching the original length'
      return self.train_graphs[item], self.features[self.train_mask_list[
          item]], self.train_labels[item], is_labeled, labeled_weights
    if self.mode == 'valid':
      return self.valid_graphs[item], self.features[
          self.valid_mask_list[item]], self.valid_labels[item]
    if self.mode == 'test':
      return self.test_graphs[item], self.features[
          self.test_mask_list[item]], self.test_labels[item]
