"""A pytorch implementation of SparseMax
Code was taken from https://github.com/OpenNMT/OpenNMT-py"""

from torch.autograd import Function
import torch.nn as nn
import torch as th

from dgl.function import TargetCode
from dgl.base import ALL, is_all
from dgl import backend as F
from dgl import utils
from dgl import backend as F
from dgl import utils
from dgl import function as fn
import pdb
# from dgl_extensions.gat_utils import save_pickle

import os
import pickle
def save_pickle(cache_file, data):
    with open(cache_file, 'wb') as f: pickle.dump(data, f)
def read_pickle(cache_file):
    with open(cache_file, 'rb') as f: data = pickle.load(f)
    return data

# from gat_utils import read_pickle
# Here we adapt Sparsemax on edges for GAT
def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = th.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)
# __all__ = ["edge_softmax"]
def _threshold_and_support(input, dim=1):
    """Sparsemax building block: compute the threshold
    - Use this directly on input = [a0, a1, a2...]
    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """

    input_srt, _ = th.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size



class EdgeSparsemax(th.autograd.Function):
    """Apply sparsemax over signals of incoming edges."""
    lambda_sparsemax = 0
    @staticmethod
    def forward(ctx, g, score, eids):
        '''
        edge sends (score-max) to nodes.mailbox
        nodes compute tau, support size and max.

        Todo:
            - Check apply_edges-interface
        '''
        if not is_all(eids):
            g = g.edge_subgraph(eids.long())
        # Store score in edge
        lamb = 1 - EdgeSparsemax.lambda_sparsemax
        # assert False
        g.edata['score'] = score / lamb
        # Comptue max(score), store in node
        g.update_all(fn.copy_e('score', 'm'), fn.max('m', 'tmp'))
        # Compute score - max_score
        g.apply_edges(fn.e_sub_v('score', 'tmp', 'norm_score'))
        g.edata.pop('score')
        g.ndata.pop('tmp')
        g.update_all(message_func=fn.copy_e('norm_score', 'norm_score'),
                     reduce_func=reduce_tau_support_size)

        g.apply_edges(edge_clamp)

        g.edata.pop('norm_score')
        sparse_outputs = g.edata.pop('out')
        support_size = g.dstdata.pop('support_size')

        ctx.backward_cache = (g, sparse_outputs, support_size)
        # ctx.dim = 0
        # pdb.set_trace()
        return sparse_outputs

    @staticmethod
    def backward(ctx, grad_out):
        """
        g, out = ctx.backward_cache
        grad_out = dgl.EData(g, grad_out)
        out = dgl.EData(g, out)
        sds = out * grad_out  # type dgl.EData
        sds_sum = sds.dst_sum()  # type dgl.NData
        grad_score = sds - sds * sds_sum  # multiple expressions
        return grad_score.data
        """
        # pdb.set_trace()
        g, out, supp_size = ctx.backward_cache
        ctx.backward_cache = None
        # dim = ctx.dim
        dim = 1
        # pdb.set_trace()

        grad_score = grad_out.clone()
        # grad_score = grad_out
        grad_score[out == 0] = 0

        # g.edata['out'] = out
        g.edata['grad_score'] = grad_score
        g.dstdata['supp_size'] = supp_size
        # g.edata['out'] = out

        # g.update_all(copy_e_to_v, reduce_sum)
        g.update_all(fn.copy_e('grad_score', 'grad_score'), reduce_sum)
        g.apply_edges(fn.e_sub_v('grad_score', 'v_hat', 'grad_input_minus_vhat'))
        grad_input_minus_vhat = g.edata.pop('grad_input_minus_vhat')
        grad_input = th.where(out != 0, grad_input_minus_vhat, grad_score)

        # assert False, 'grad is not computed correctly '
        # clean the storage
        g.dstdata.pop('v_hat')
        g.dstdata.pop('supp_size')
        g.edata.pop('grad_score')
        # assert False
        # pdb.set_trace()
        # data = [supp_size]
        # cache_file = 'all_remaining.pkl'
        # save_pickle(cache_file, data)
        # assert False, 'should stop here first'
        # pdb.set_trace()
        return None, grad_input, None


        # g.update_all(fn.copy_e('grad_score', 'msg'), reduce_sum)
        # pdb.set_trace()
        # v_hat = g.dstdata.pop('accum') / supp_size[:, 0, :, :]
        # v_hat = v_hat.unsqueeze(dim)
        # g.dstdata['v_hat'] = v_hat

        # g.apply_edges(our_e_sub_v)
        # inf. Div by 0

# def copy_e_to_v(edges):
#     out = edges.data['out']
#     grad_score = edges.data['grad_score']
#     return {
#         'out': out,
#         'grad_score': grad_score
#     }

def reduce_sum(nodes):
    msg = nodes.mailbox['grad_score']
    supp_size = nodes.data['supp_size']
    dim = 1
    msg_sum = th.sum(msg, dim=1)
    supp_size = supp_size.view(msg_sum.shape)
    # v_hat = msg_sum / supp_size
    # print('msg: {}'.format(msg.shape))
    # print('msg_sum: {}'.format(msg_sum.shape))
    # print('supp_size: {}'.format(supp_size.shape))
    # print()
    v_hat = msg_sum / supp_size
    # v_hat = v_hat.unsqueeze(dim)

    # if msg.shape[1] == 5 and msg.shape[2]>1:
    #     out = nodes.mailbox['out']
    #     # pdb.set_trace()
    #     non_zero_indices = msg.sum(dim=1).nonzero(as_tuple=True)
    #     nodes_ids = nodes.nodes()[non_zero_indices[0]]
    #     data = [msg, out, supp_size, v_hat, nodes_ids]
    #     cache_file = 'all.pkl'
    #     save_pickle(cache_file, data)

    results = {
        'v_hat': msg_sum
    }
    return results


def reduce_tau_support_size(nodes):
    tau, support_size = _threshold_and_support(nodes.mailbox['norm_score'], dim=1)
    # pdb.set_trace()
    # assert len(tau) == 1, 'should output one single number only'
    results = {
        'tau': tau, #.squeeze(),
        'support_size': support_size #.squeeze()
    }
    return results


def edge_clamp(edges):
    tau = edges.dst['tau']
    input = edges.data['norm_score']
    # pdb.set_trace()
    output = th.clamp(input - tau.view(input.shape), min=0)
    results = {
        'out': output
    }
    return results

def edge_sparsemax(graph, logits, eids=ALL):
    return EdgeSparsemax.apply(graph, logits, eids)



# https://github.com/adrian-lison/gnn-community-detection/blob/master/code/Sparsemax.py



# def _threshold_and_support(input, dim=0):
#     """Sparsemax building block: compute the threshold
#     - Use this directly on input = [a0, a1, a2...]
#     Args:
#         input: any dimension
#         dim: dimension along which to apply the sparsemax
#     Returns:
#         the threshold value
#     """

#     input_srt, _ = th.sort(input, descending=True, dim=dim)
#     input_cumsum = input_srt.cumsum(dim) - 1
#     rhos = _make_ix_like(input, dim)
#     support = rhos * input_srt > input_cumsum

#     support_size = support.sum(dim=dim).unsqueeze(dim)
#     tau = input_cumsum.gather(dim, support_size - 1)
#     tau /= support_size.to(input.dtype)
#     return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = th.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = th.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


#########################
class SparsemaxFunction(th.autograd.Function):

    @staticmethod
    def forward(self, z, dim=-1):
        z = z.transpose(dim, -1)
        sorted_z, _ = z.sort(-1, descending=True)
        k_z = (1 + th.arange(1., 1 + z.shape[-1]) * sorted_z > sorted_z.cumsum(-1)).sum(-1, keepdim=True) + 1
        mask = (k_z >= th.arange(1, z.shape[-1] + 1).repeat(z.shape[:-1] + (1, )))
        t_z = ((mask.float() * sorted_z).sum(-1, keepdim=True) - 1) / k_z.float()
        out = th.relu(z - t_z)
        self.save_for_backward(out, th.tensor(dim))
        return out.transpose(dim, -1)

    @staticmethod
    def backward(self, grad_output):
        dim = self.saved_tensors[1]
        s = (self.saved_tensors[0] > 0).float()
        v = (s * grad_output).sum(dim, keepdim=True) / s.sum(dim, keepdim=True)
        grad = s * (grad_output - v)
        return grad
    
sparsemax_2 = SparsemaxFunction.apply
    
class Sparsemax_2(th.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, z):
        return SparsemaxFunction.apply(z)