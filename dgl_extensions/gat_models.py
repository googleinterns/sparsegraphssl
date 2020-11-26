import torch.nn as nn
from EvolveGCNExtension.SparseGATConv import SparseGATConv
from EvolveGCNExtension.SparseGATEdgeFeatConv import SparseGATEdgeFeatConv
from EvolveGCNExtension.SoftSparseGATConv import SoftSparseGATConv
from EvolveGCNExtension.LearnableSoftSparseGATConv import LearnableSoftSparseGATConv
from EvolveGCNExtension.UnsupSparseGATConv import UnsupSparseGATConv


# from EvolveGCNExtension.models_evolveGCN import GAT
from dgl.nn.pytorch import edge_softmax, GATConv
from gat import GAT as OrigGAT


class GAT(OrigGAT):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 **kwargs):
        assert num_layers >= 2, 'GAT support only min. 2 Layers'
        num_layers = num_layers - 1
        super(GAT, self).__init__(g,
                                  num_layers,
                                  in_dim,
                                  num_hidden,
                                  num_classes,
                                  heads,
                                  activation,
                                  feat_drop,
                                  attn_drop,
                                  negative_slope,
                                  residual)
    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class SparseGAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 p_norm=None,
                 # use_cuda=False,
                 conv=SparseGATConv,
                 softmax='edge_softmax',
                 lambda_sparsemax=None,
                 alpha_sparsemax=None,
                 unsup_sparsemax_weights=None,
                 **kwargs):
        # (self,  gcn_args, gat_parameters, activation,  p_norm=None, seed=1234):
        # super(SparseGAT, self).__init__(gcn_args, gat_parameters, activation, seed, conv=SparseGATConv)

        super(SparseGAT, self).__init__()

        # conv = SparseGATConv
        self.g = g
        self.p_norm = p_norm
        assert num_layers >=2, 'GAT support only min. 3 Layers'
        self.num_layers = num_layers-1 
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        # input projection (no residual)
        self.gat_layers.append(conv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, self.p_norm, softmax, lambda_sparsemax, alpha_sparsemax, unsup_sparsemax_weights))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_feats = num_hidden * num_heads
            self.gat_layers.append(conv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, self.p_norm, softmax, lambda_sparsemax, alpha_sparsemax, unsup_sparsemax_weights))
        # output projection
        self.gat_layers.append(conv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, self.p_norm, softmax, lambda_sparsemax, alpha_sparsemax, unsup_sparsemax_weights))

        # if use_cuda:
        #     ## converte to cuda layers
        #     for i, layer in enumerate(self.gat_layers):
        #         self.gat_layers[i] = layer.cuda()
    def forward(self, inputs):
        h = inputs
        #pdb.set_trace()
        attention_weights = []
        sqrt_norm_all = []
        for l in range(self.num_layers):
            h, a_weights, sqrt_norm = self.gat_layers[l](self.g, h)
            h = h.flatten(1)
            attention_weights.append(a_weights)
            sqrt_norm_all.append(sqrt_norm)
        # output projection
        logits, a_weights, sqrt_norm = self.gat_layers[-1](self.g, h)
        logits = logits.mean(1)
        attention_weights.append(a_weights)
        sqrt_norm_all.append(sqrt_norm)

        return logits, attention_weights, sqrt_norm_all



class SparseGATEdgeFeat(SparseGAT):
    def __init__(self,
                 *args, 
                 **kwargs):
        super(SparseGATEdgeFeat, self).__init__(*args,
                                                conv=SparseGATEdgeFeatConv,
                                                **kwargs
                                                )


class SoftSparseGAT(SparseGAT):
    def __init__(self,
                 *args, 
                 **kwargs):
        super(SoftSparseGAT, self).__init__(*args,
                                            conv=SoftSparseGATConv,
                                            **kwargs
                                            )


class LearnableSoftSparseGAT(SparseGAT):
    def __init__(self,
                 *args, 
                 **kwargs):
        super(LearnableSoftSparseGAT, self).__init__(*args,
                                                     conv=LearnableSoftSparseGATConv,
                                                     **kwargs
                                                     )


class UnsupSparseGAT(SparseGAT):
    def __init__(self,
                 *args, 
                 **kwargs):
        super(UnsupSparseGAT, self).__init__(*args,
                                            conv=UnsupSparseGATConv,
                                            **kwargs
                                            )