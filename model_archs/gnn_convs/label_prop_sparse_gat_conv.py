"""Torch modules for graph attention networks(GAT)."""
# Lint as  python3

from dgl import function as fn
from dgl.nn.pytorch import GATConv
from model_archs.operations.edge_sparsemax import edge_sparsemax


class LabelPropSparseGATConv(GATConv):
  r"""Enable label propagation by freezing attention_weights with sparsemax.

  In addition to the normal forward pass in a sparsemax-layer, this class stores
  the
  attention weights after each forward pass. These weights can then be reused
  for label propagation.

  """

  def __init__(
      self,
      in_feats,
      out_feats,
      num_heads,
      feat_drop=0.,
      attn_drop=0.,
      negative_slope=0.2,
      residual=False,
      activation=None,
      p_norm=0.5,
      lambda_sparsemax=None,
      test_level=None,
      **kwargs,
  ):
    r"""Enable label propagation by freezing attention_weights with sparsemax.

    In addition to the normal forward pass in a sparsemax-layer, this class
    stores the attention weights after each forward pass. These weights can then
    be reused for label propagation.

    Args:
      in_feats : int, or pair of ints Input feature size.  If the layer is to be
        applied to a unidirectional bipartite graph, ``in_feats`` specifies the
        input feature size on both the source and destination nodes If a scalar
        is given, the source and destination node feature size would take
        thesame value.
      out_feats : int Output feature size.
      num_heads : int Number of heads in Multi-Head Attention.
      feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
      attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
      negative_slope : float, optional LeakyReLU angle of negative slope.
      residual : bool, optional If True, use residual connection.
      activation : callable activation function/layer or None, optional
      p_norm : the L_p norm to be computed during the forward pass.
      lambda_sparsemax : lambda to anneal the sparsemax operation
      test_level : Testing mode.
      **kwargs: ignored keyword arguments. Added for compatiblity reasons.
    """
    super(LabelPropSparseGATConv,
          self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop,
                         negative_slope, residual, activation)

    # assert unsupervised is not None
    self.p_norm = p_norm
    self.lambda_sparsemax = lambda_sparsemax
    self.test_level = test_level

  def single_forward_pass(self, graph, feat, reuse_att=False):
    r"""Compute the forward pass of a sparse graph attention network layer.

    Compute the additive attetnion and save the attention for a potential
    reuse.

    Args:
      graph : DGLGraph The graph.
      feat : torch.Tensor or pair of torch.Tensor
        If a torch.Tensor is given, the input feature of shape :math:`(N,
          D_{in})` where :math:`D_{in}` is size of input feature, :math:`N` is
            the number of nodes. If a pair of torch.Tensor is given, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
              :math:`(N_{out}, D_{in_{dst}})`.
      reuse_att: if True do not compute the attention weights again. Can be used
        to compute label propagation

    Returns:
      rst: torch.Tensor
        The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
        is the number of heads, and :math:`D_{out}` is size of output feature.
    """

    with graph.local_scope():
      if feat.is_leaf:
        # input feature requires extra grads
        feat.requires_grad = True
      h_src = h_dst = self.feat_drop(feat)

      if not reuse_att:
        feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads,
                                                  self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i
        # and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))

        one_minus_lambda_sparsemax = (1 - self.lambda_sparsemax)
        lambda_score = e / one_minus_lambda_sparsemax

        sparse_attn = edge_sparsemax(graph, lambda_score)
        graph.edata['a'] = self.attn_drop(sparse_attn)

        self.attention_weights = graph.edata['a']
      else:
        assert self.attention_weights is not None, (
            'attention weights are missing')
        graph.srcdata.update({'ft': h_src})
        graph.edata['a'] = self.attention_weights.detach()
      # message passing to combine ft based on attention weights
      graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
      rst = graph.dstdata['ft']
      # residual
      if self.res_fc is not None:
        resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
        rst = rst + resval
      # activation
      if self.activation:
        rst = self.activation(rst)
      return rst

  def forward(self, graph, feat, reuse_att=False):
    rst = self.single_forward_pass(graph, feat, reuse_att=reuse_att)
    return rst
