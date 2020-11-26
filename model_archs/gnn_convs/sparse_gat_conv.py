"""SparseMax-Attention-Layer."""
# Lint as  python3

from dgl import function as fn
from dgl.nn.pytorch import GATConv
from model_archs.operations.edge_sparsemax import edge_sparsemax
# from model_archs.gat_utils import compute_att_wrt_h


class SparseGATConv(GATConv):
  r"""Apply SparseMax to incoming input signals.

  A graph convolution layer which allow to apply SparseMax on the edges

  """

  def __init__(self,
               in_feats,
               out_feats,
               num_heads,
               feat_drop=0.,
               attn_drop=0.,
               negative_slope=0.2,
               residual=False,
               activation=None,
               lambda_sparsemax=None,
               **kwargs):
    super(SparseGATConv,
          self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop,
                         negative_slope, residual, activation)
    self.lambda_sparsemax = lambda_sparsemax

  def forward(self, graph, feat):
    r"""Compute sparsemax attention..

    Creates a local copy of the graph and compute the forward pass of the
    sparsemax operation.

    Args:
      graph : DGLGraph The graph.
      feat : torch.Tensor or pair of torch.Tensor
        If a torch.Tensor is given, the input feature of shape :math:`(N,
          D_{in})` where :math:`D_{in}` is size of input feature, :math:`N` is
            the number of nodes. If a pair of torch.Tensor is given, the pair
            must
          contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.

    Returns:
      torch.Tensor
        The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
        is the number of heads, and :math:`D_{out}` is size of output feature.
    """
    graph = graph.local_var()

    if isinstance(feat, tuple):
      h_src = self.feat_drop(feat[0])
      h_dst = self.feat_drop(feat[1])
      feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
      feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
    else:
      h_src = h_dst = self.feat_drop(feat)
      feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads,
                                                self._out_feats)
    el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
    er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
    graph.srcdata.update({'ft': feat_src, 'el': el})
    graph.dstdata.update({'er': er})
    # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
    e = self.leaky_relu(graph.edata.pop('e'))

    one_minus_lambda_sparsemax = (1 - self.lambda_sparsemax)
    lambda_score = e / one_minus_lambda_sparsemax

    sparse_attn = edge_sparsemax(graph, lambda_score)
    graph.edata['a'] = self.attn_drop(sparse_attn)
    self.attention_weights = graph.edata['a']

    # self.att_wrt_h = compute_att_wrt_h(self.attention_weights, self.fc.weight)

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
