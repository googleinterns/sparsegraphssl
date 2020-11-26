"""Torch modules for graph attention networks(GAT)."""
# Lint as  python3

from dgl.nn.pytorch import GATConv
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax


class CustomGATConv(GATConv):
  """Wrapper of GATConv to ignore named arguments

  This class wraps GATConv to ignore named arguments

  """

  def __init__(
      self,
      *args,
      **kwargs,
  ):
    """Wrapper of GATConv to ignore named arguments"""
    super(CustomGATConv, self).__init__(*args)

  def forward(self, graph, feat):
    """Compute graph attention network layer.

    Args: ----------
      graph : DGLGraph The graph.
      feat : torch.Tensor or pair of torch.Tensor
          If a torch.Tensor is given, the input feature of shape :math:`(N,
          D_{in})` where :math:`D_{in}` is size of input feature, :math:`N` is
          the number of nodes. If a pair of torch.Tensor is given, the pair must
          contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
          :math:`(N_{out}, D_{in_{dst}})`.  Returns ------- torch.Tensor
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
    # compute softmax
    graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
    self.attention_weights = graph.edata['a']

    # message passing
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
    rst = graph.dstdata['ft']
    if self.res_fc is not None:
      resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
      rst = rst + resval
    if self.activation:
      rst = self.activation(rst)
    return rst
