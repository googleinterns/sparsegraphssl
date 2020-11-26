"""Cora, citeseer, pubmed dataset.

A third-party implementation of citation graph dataset from dgl.4.0.1 post. This
file is needed to retain compatibility.

"""
from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
from dgl import DGLGraph
from functools import wraps

_urls = {
    'cora_v2': 'dataset/cora_v2.zip',
    'citeseer': 'dataset/citeseer.zip',
    'pubmed': 'dataset/pubmed.zip',
    'cora_binary': 'dataset/cora_binary.zip',
}


def _pickle_load(pkl_file):
  """Summary

  Parameters
  ----------
  pkl_file : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  if sys.version_info > (3, 0):
    return pkl.load(pkl_file, encoding='latin1')
  else:
    return pkl.load(pkl_file)


def retry_method_with_fix(fix_method):
  """Decorator that executes a fix method before retrying again when the decorated method

  fails once with any exception.

  If the decorated method fails again, the execution fails with that
  exception.

  Notes
  -----
  This decorator only works on class methods, and the fix function must also
  be a class method.
  It would not work on functions.

  Parameters
  ----------
  fix_method : TYPE
      Description

  Deleted Parameters
  ------------------
  fix_func : callable
      The fix method to execute.  It should not accept any arguments.  Its
      return values are
      ignored.

  Returns
  -------
  TYPE
      Description
  """

  def _creator(func):
    """Summary

    Parameters
    ----------
    func : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
      """Summary

      Parameters
      ----------
      *args
          Description
      **kwargs
          Description

      Returns
      -------
      TYPE
          Description
      """
      # pylint: disable=W0703,bare-except
      try:
        return func(self, *args, **kwargs)
      except:
        fix_method(self)
        return func(self, *args, **kwargs)

    return wrapper

  return _creator


class CitationGraphDataset(object):
  """The citation graph dataset, including cora, citeseer and pubmeb.

  Nodes mean authors and edges mean citation relationships.

  Parameters
  ----------
  name : str
    name can be 'cora', 'citeseer' or 'pubmed'.

  Attributes
  ----------
  dir : TYPE
      Description
  features : TYPE
      Description
  graph : TYPE
      Description
  labels : TYPE
      Description
  name : TYPE
      Description
  num_labels : TYPE
      Description
  onehot_labels : TYPE
      Description
  test_mask : TYPE
      Description
  train_mask : TYPE
      Description
  val_mask : TYPE
      Description
  zip_file_path : TYPE
      Description
  """

  def __init__(self, name):
    """Summary

    Parameters
    ----------
    name : TYPE
        Description
    """
    assert name.lower() in ['cora', 'citeseer', 'pubmed']

    # Previously we use the pre-processing in pygcn (https://github.com/tkipf/pygcn)
    # for Cora, which is slightly different from the one used in the GCN paper
    if name.lower() == 'cora':
      name = 'cora_v2'

    self.name = name
    self.dir = get_download_dir()
    self.zip_file_path = '{}/{}.zip'.format(self.dir, name)
    self._load()

  def _download_and_extract(self):
    """Summary"""
    download(_get_dgl_url(_urls[self.name]), path=self.zip_file_path)
    extract_archive(self.zip_file_path, '{}/{}'.format(self.dir, self.name))

  @retry_method_with_fix(_download_and_extract)
  def _load(self):
    """Loads input data from gcn/data directory

    ind.name.x => the feature vectors of the training instances as
    scipy.sparse.csr.csr_matrix object;
    ind.name.tx => the feature vectors of the test instances as
    scipy.sparse.csr.csr_matrix object;
    ind.name.allx => the feature vectors of both labeled and unlabeled
    training instances
        (a superset of ind.name.x) as scipy.sparse.csr.csr_matrix object;
    ind.name.y => the one-hot labels of the labeled training instances as
    numpy.ndarray object;
    ind.name.ty => the one-hot labels of the test instances as numpy.ndarray
    object;
    ind.name.ally => the labels for instances in ind.name.allx as
    numpy.ndarray object;
    ind.name.graph => a dict in the format {index:
    [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.name.test.index => the indices of test instances in graph, for the
    inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param name: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    root = '{}/{}'.format(self.dir, self.name)
    objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(objnames)):
      with open('{}/ind.{}.{}'.format(root, self.name, objnames[i]), 'rb') as f:
        objects.append(_pickle_load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file('{}/ind.{}.test.index'.format(
        root, self.name))
    test_idx_range = np.sort(test_idx_reorder)

    if self.name == 'citeseer':
      # Fix citeseer dataset (there are some isolated nodes in the graph)
      # Find isolated nodes, add them as zero-vecs into the right position
      test_idx_range_full = range(
          min(test_idx_reorder),
          max(test_idx_reorder) + 1)
      tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
      tx_extended[test_idx_range - min(test_idx_range), :] = tx
      tx = tx_extended
      ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
      ty_extended[test_idx_range - min(test_idx_range), :] = ty
      ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    graph = nx.DiGraph(nx.from_dict_of_lists(graph))

    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
    labels = np.argmax(onehot_labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = _sample_mask(idx_train, labels.shape[0])
    val_mask = _sample_mask(idx_val, labels.shape[0])
    test_mask = _sample_mask(idx_test, labels.shape[0])

    self.graph = graph
    self.features = _preprocess_features(features)
    self.labels = labels
    self.onehot_labels = onehot_labels
    self.num_labels = onehot_labels.shape[1]
    self.train_mask = train_mask
    self.val_mask = val_mask
    self.test_mask = test_mask

    print('Finished data loading and preprocessing.')
    print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
    print('  NumEdges: {}'.format(self.graph.number_of_edges()))
    print('  NumFeats: {}'.format(self.features.shape[1]))
    print('  NumClasses: {}'.format(self.num_labels))
    print('  NumTrainingSamples: {}'.format(
        len(np.nonzero(self.train_mask)[0])))
    print('  NumValidationSamples: {}'.format(
        len(np.nonzero(self.val_mask)[0])))
    print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

  def __getitem__(self, idx):
    """Summary

    Parameters
    ----------
    idx : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    assert idx == 0, 'This dataset has only one graph'
    g = DGLGraph(self.graph)
    g.ndata['train_mask'] = self.train_mask
    g.ndata['val_mask'] = self.val_mask
    g.ndata['test_mask'] = self.test_mask
    g.ndata['label'] = self.labels
    g.ndata['feat'] = self.features
    return g

  def __len__(self):
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    return 1


def _preprocess_features(features):
  """Row-normalize feature matrix and convert to tuple representation

  Parameters
  ----------
  features : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  rowsum = np.asarray(features.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  features = r_mat_inv.dot(features)
  return np.asarray(features.todense())


def _parse_index_file(filename):
  """Parse index file.

  Parameters
  ----------
  filename : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  index = []
  for line in open(filename):
    index.append(int(line.strip()))
  return index


def _sample_mask(idx, l):
  """Create mask.

  Parameters
  ----------
  idx : TYPE
      Description
  l : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  mask = np.zeros(l)
  mask[idx] = 1
  return mask


def load_cora():
  """Summary

  Returns
  -------
  TYPE
      Description
  """
  data = CitationGraphDataset('cora')
  return data


def load_citeseer():
  """Summary

  Returns
  -------
  TYPE
      Description
  """
  data = CitationGraphDataset('citeseer')
  return data


def load_pubmed():
  """Summary

  Returns
  -------
  TYPE
      Description
  """
  data = CitationGraphDataset('pubmed')
  return data


def get_gnp_generator(args):
  """Summary

  Parameters
  ----------
  args : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  n = args.syn_gnp_n
  p = (2 * np.log(n) / n) if args.syn_gnp_p == 0. else args.syn_gnp_p

  def _gen(seed):
    """Summary

    Parameters
    ----------
    seed : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return nx.fast_gnp_random_graph(n, p, seed, True)

  return _gen


class ScipyGraph(object):
  """A simple graph object that uses scipy matrix."""

  def __init__(self, mat):
    """Summary

    Parameters
    ----------
    mat : TYPE
        Description
    """
    self._mat = mat

  def get_graph(self):
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    return self._mat

  def number_of_nodes(self):
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    return self._mat.shape[0]

  def number_of_edges(self):
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    return self._mat.getnnz()


def get_scipy_generator(args):
  """Summary

  Parameters
  ----------
  args : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  n = args.syn_gnp_n
  p = (2 * np.log(n) / n) if args.syn_gnp_p == 0. else args.syn_gnp_p

  def _gen(seed):
    """Summary

    Parameters
    ----------
    seed : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return ScipyGraph(sp.random(n, n, p, format='coo'))

  return _gen


def register_args(parser):
  """Summary

  Parameters
  ----------
  parser : TYPE
      Description
  """
  # Args for synthetic graphs.
  parser.add_argument(
      '--syn-type',
      type=str,
      default='gnp',
      help='Type of the synthetic graph generator')
  parser.add_argument(
      '--syn-nfeats', type=int, default=500, help='Number of node features')
  parser.add_argument(
      '--syn-nclasses', type=int, default=10, help='Number of output classes')
  parser.add_argument(
      '--syn-train-ratio',
      type=float,
      default=.1,
      help='Ratio of training nodes')
  parser.add_argument(
      '--syn-val-ratio',
      type=float,
      default=.2,
      help='Ratio of validation nodes')
  parser.add_argument(
      '--syn-test-ratio', type=float, default=.5, help='Ratio of testing nodes')
  # Args for GNP generator
  parser.add_argument(
      '--syn-gnp-n', type=int, default=1000, help='n in gnp random graph')
  parser.add_argument(
      '--syn-gnp-p', type=float, default=0.0, help='p in gnp random graph')
  parser.add_argument('--syn-seed', type=int, default=42, help='random seed')


def _normalize(mx):
  """Row-normalize sparse matrix

  Parameters
  ----------
  mx : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  rowsum = np.asarray(mx.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx


def _encode_onehot(labels):
  """Summary

  Parameters
  ----------
  labels : TYPE
      Description

  Returns
  -------
  TYPE
      Description
  """
  classes = list(sorted(set(labels)))
  classes_dict = {
      c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
  }
  labels_onehot = np.asarray(
      list(map(classes_dict.get, labels)), dtype=np.int32)
  return labels_onehot
