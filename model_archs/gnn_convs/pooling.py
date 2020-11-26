from torch import nn
from torch.nn import functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape


class Pooling(nn.Module):
  r"""Pooling layer of GraphSAGE layer from paper `Inductive Representation Learning on

    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied on a unidirectional bipartite graph,
        ``in_feats``
        specifies the input feature size on both the source and destination
        nodes.  If
        a scalar is given, the source and destination node feature size would
        take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and
        destination nodes
        are required to be the same.
    out_feats : int
        Output feature size. When used with Label progation, use out_feats =
        n_class.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node
        features.
        Default: ``None``.
    """

  def __init__(
      self,
      in_feats,
      out_feats,
      aggregator_type,
      feat_drop=0.,
      norm=None,
      activation=None,
      pooling_residual=None,
  ):
    super(Pooling, self).__init__()

    self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
    self._out_feats = out_feats
    self._aggre_type = aggregator_type
    self.norm = norm
    self.feat_drop = nn.Dropout(feat_drop)
    self.activation = activation
    self.pooling_residual = pooling_residual

  def reset_parameters(self):
    """Reinitialize learnable parameters."""
    if self.transform == 'vector':
      nn.init.constant(self.vec_self, 1)
      nn.init.constant(self.vec_neigh, 1)

  def forward(self, graph, feat, weights=None):
    r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N,
            D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of
            nodes.
            If a pair of torch.Tensor is given, the pair must contain two
            tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}`
            is size of output feature.
        """
    with graph.local_scope():
      if isinstance(feat, tuple):
        feat_src = self.feat_drop(feat[0])
        feat_dst = self.feat_drop(feat[1])
      else:
        feat_src = feat_dst = self.feat_drop(feat)

      if self._aggre_type == 'mean':
        graph.srcdata['h'] = feat_src
        graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
      elif self._aggre_type == 'gcn':
        check_eq_shape(feat)
        graph.srcdata['h'] = feat_src
        graph.dstdata['h'] = feat_dst  # same as above if homogeneous
        graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
        # divide in_degrees
        degs = graph.in_degrees().to(feat_dst)
        h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (
            degs.unsqueeze(-1) + 1)
      elif self._aggre_type == 'pool':
        graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
        graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
      elif self._aggre_type == 'lstm':
        graph.srcdata['h'] = feat_src
        graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
        h_neigh = graph.dstdata['neigh']
      elif self._aggre_type == 'weighted_pooling':
        assert weights is not None
        graph.edata['w'] = weights
        graph.srcdata['h'] = feat_src
        graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
      else:
        raise KeyError('Aggregator type {} not recognized.'.format(
            self._aggre_type))
      if self.pooling_residual:
        h_neigh = h_neigh + feat_dst

    return h_neigh
