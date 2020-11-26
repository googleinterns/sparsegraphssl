import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import edge_softmax, GATConv
import pdb
from dgl_extensions.sparsemax import EdgeSparsemax
from dgl_extensions.sparsemax import edge_sparsemax
from EvolveGCNExtension.SparseGATConv import SparseGATConv
from dgl.base import ALL, is_all
from EvolveGCNExtension.SparseGATConv import att_sqrt
from EvolveGCNExtension.SparseGATConv import reduce_sum_sqr




class UnsupSparseGATConv(SparseGATConv):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.
        - return the attention weigths additionally
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
                 p_norm=0.5,
                 softmax='edge_softmax',
                 lambda_sparsemax=None,
                 alpha_sparsemax=None,
                 unsup_sparsemax_weights=None,
                 **kwargs):
        super(UnsupSparseGATConv, self).__init__(
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 activation)
        self.p_norm = p_norm
        self.alpha_sparsemax = alpha_sparsemax
        assert alpha_sparsemax is not None
        # if softmax == 'edge_softmax':
        #     self.softmax_fn = edge_softmax
        # elif softmax == 'sparse_max':
        def edge_sparsemax_lambda(graph, logits):
            EdgeSparsemax.lambda_sparsemax = lambda_sparsemax
            eids = ALL
            return EdgeSparsemax.apply(graph, logits, eids)
        self.edge_sparsemax = edge_sparsemax_lambda
        self.unsup_sparsemax_weights = unsup_sparsemax_weights
        # else:
        #     raise NotImplementedError


    def forward(self, graph, feat):
        r"""Compute graph attention network layer.
        unlabeled samples receives a sparsemax weight = 1 --> use sparsemax
        labeled samples receives sparsemax weight = 0.
        - Valid and test always get weigths = 1.
        sparsemax_weights comes from data setting.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
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
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        # linear compbination:

        # flattened_feats = feat_src.view(feat_src.shape[0], self._num_heads * self._out_feats)

        # alpha_sparsemax = self.sigmoid((self.attn_alpha * feat_src).sum(dim=-1).sum(dim=-1)).unsqueeze(-1).unsqueeze(-1)
        # alpha_sparsemax = self.sigmoid((self.attn_alpha * feat_src).sum(dim=-1)).unsqueeze(-1)


        # alpha_sparsemax = self.sigmoid(self.fc_alpha(flattened_feats))#.squeeze()


        # combined_results = alpha_sparsemax * sparsemax_results + (1-alpha_sparsemax) * softmax_results
        # Combine softmax and sparsemax individually.
        graph.edata['sparsemax'] = self.edge_sparsemax(graph, e)
        graph.ndata['unlabeled_weights'] = 1 - self.unsup_sparsemax_weights
        graph.apply_edges(fn.e_mul_u('sparsemax', 'unlabeled_weights', 'sparsemax_results'))

        graph.edata['softmax'] = edge_softmax(graph, e)
        graph.ndata['labeled_weights'] = self.unsup_sparsemax_weights
        graph.apply_edges(fn.e_mul_u('softmax', 'labeled_weights', 'softmax_results'))


        masked_sparsemax_results = graph.edata.pop('sparsemax_results')
        masked_softmax_results = graph.edata.pop('softmax_results')
        combined_results = masked_softmax_results + masked_sparsemax_results
        # print(masked_softmax_results[400])
        # print(masked_sparsemax_results[400])
        # assert combined_results[0].sum() == masked_softmax_results[0].sum() or combined_results[0].sum() == masked_sparsemax_results[0].sum(), 'Make sure results are either or only.'
        # pdb.set_trace()
        # pdb.set_trace()

        graph.edata.pop('sparsemax')
        graph.ndata.pop('unlabeled_weights')
        graph.edata.pop('softmax')
        graph.ndata.pop('labeled_weights')
        # clean graph:



        graph.edata['a'] = self.attn_drop(combined_results)
        # graph.edata['a'] = edge_softmax(graph, e)
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # this support custom norms such as 0.5
        # graph.update_all(message_func=att_sqrt,
        # reduce_func=fn.sum('sqrt_alpha', 'sum_sqrt_alpha'))
        p_norm = self.p_norm
        if p_norm is not None:
            graph.update_all(message_func= lambda edges: att_sqrt(p_norm, edges),
                             reduce_func= lambda edges: reduce_sum_sqr(p_norm, edges))
            normed_attentions = graph.ndata['sum_sqrt_alpha']
        else:
            normed_attentions = None
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        # pdb.set_trace()

        return rst, graph.edata['a'], normed_attentions