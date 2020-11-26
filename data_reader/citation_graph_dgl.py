"""CitationGraphDataset for custom usage."""
from data_reader.citation_graph import CitationGraphDataset
from data_reader.citation_graph import _sample_mask
import numpy as np
from data_reader.data_utils import get_full_train_mask
from data_reader.data_utils import sample_train_mask
from data_reader.data_utils import create_graph
from data_reader.data_utils import add_noise_to_graph
from data_reader.data_utils import set_seed


class CitationGraphDGLDataset(CitationGraphDataset):
  """A citation graph dataset to support custom usage.

  Attributes:
      args (namespace): all arguments.
      dataset_name (str): name of the dataset.
      g (DGLGraph): the graph.
      noise_injected_ratio (float): how much noise to inject into the training.
      noisy_edges_ids (list): which edges are noisy.
      train_mask (torch.BoolTensor): Which samples are used for training.
  """

  def __init__(self, name, args):
    """Initialize the dataset.

    Args:
        name (str): name of the dataest
        args (namespace): all arguments
    """
    self.args = args
    self.dataset_name = self.args.data
    self.noise_injected_ratio = args.noise_injected_ratio
    super(CitationGraphDGLDataset, self).__init__(name)

  def _load(self):
    """Loads and proprocess the datasets."""
    super()._load()
    if self.args.correct_data_split:
      full_train_indices = get_full_train_mask(self.train_mask, self.val_mask,
                                               self.test_mask, len(self.labels))
    sampled_train_indices = sample_train_mask(
        full_train_indices,
        self.args,
        self.labels,
        seed=self.args.repeated_runs)
    self.train_mask = _sample_mask(sampled_train_indices, len(self.labels))

    self.g = create_graph(self.graph)
    if self.noise_injected_ratio > 0:
      set_seed(1234)
      self.g, noisy_edges_ids = add_noise_to_graph(
          self.g,
          noise_ratio=self.noise_injected_ratio,
          noise_type=self.args.noise_type,
          labels=self.labels)
      self.noisy_edges_ids = noisy_edges_ids
      assert len(noisy_edges_ids) > 0
    else:
      self.noisy_edges_ids = []
    print('Finished data loading and preprocessing.')
    print('  NumNodes: {}'.format(self.g.number_of_nodes()))
    print('  NumEdges: {}'.format(self.g.number_of_edges()))
    print('  NumFeats: {}'.format(self.features.shape[1]))
    print('  NumClasses: {}'.format(self.num_labels))
    print('  NumTrainingSamples: {}'.format(
        len(np.nonzero(self.train_mask)[0])))
    print('  NumValidationSamples: {}'.format(
        len(np.nonzero(self.val_mask)[0])))
    print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))


def load_cora(args):
  """Loads cora-dataset.

  Args:
      args (namespace): all arguments

  Returns:
      data: object to have acess to load data
  """
  data = CitationGraphDGLDataset('cora', args)
  return data


def load_citeseer(args):
  """Loads citeseer dataset.

  Args:
      args (namespace): all arguments

  Returns:
      data: object to have acess to load data
  """
  data = CitationGraphDGLDataset('citeseer', args)
  return data


def load_pubmed(args):
  """Loads pubmed dtaaset.

  Args:
      args (namespace): all arguments

  Returns:
      data: object to have acess to load data
  """
  data = CitationGraphDGLDataset('pubmed', args)
  return data


def load_data(args):
  """Summary

  Args:
      args (namespace): all arguments

  Returns:
      data: object to have acess to load data

  Raises:
      ValueError: Description
  """
  if args.data == 'cora':
    return load_cora(args)
  elif args.data == 'citeseer':
    return load_citeseer(args)
  elif args.data == 'pubmed':
    return load_pubmed(args)
  else:
    raise ValueError('Unknown data: {}'.format(args.data))
