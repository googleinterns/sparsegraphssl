from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
import scipy.io as sio

import networkx as nx
from networkx.readwrite import json_graph
import dgl.data.citation_graph as citegrh
from dgl.data.citation_graph import CitationGraphDataset
from dgl.data.citation_graph import CoraBinary
import utils as u
import os 
import torch
import numpy as np
from dgl import DGLGraph
import warnings
from dgl_extensions.gat_utils import get_full_train_mask
from dgl_extensions.gat_utils import sample_train_mask
from dgl import DGLGraph
from data_reader.citation_graph_extended import handle_nhops
from data_reader.citation_graph import _sample_mask

WALK_LEN=5  
N_WALKS=50
# net_rur, net_rtr, net_rsr: three sparse matrices representing three homo-graphs defined in GraphConsis paper;

from utils import get_sample_args
from tqdm import tqdm
from dgl_extensions.gat_utils import load_yelp
from dgl_extensions.gat_utils import split_yelp_data
import dgl_extensions.gat_utils as gat_utils
import pdb

class LowLabelYelpChiDataset(object):
    """A toy Protein-Protein Interaction network dataset.

    Adapted from https://github.com/williamleif/GraphSAGE/tree/master/example_data.

    The dataset contains 24 graphs. The average number of nodes per graph
    is 2372. Each node has 50 features and 121 labels.

    We use 20 graphs for training, 2 for validation and 2 for testing.
    """
    def __init__(self, args, relations):
        """Initialize the dataset.

        Paramters
        ---------
        mode : str
            ('train', 'valid', 'test').
        """
        self.args = args
        self.relations = relations
        # super(LowLabelEllDataset, self).__init__(mode)
        self._prepare_params(relations)
        self._load()


    def _prepare_params(self, relations):
        '''
        create masks for train, val, test
        add labels and features
        '''
        G, adj_mat, features, labels = load_yelp(relations=relations)
        self.features = features
        self.labels = np.array(labels)
        self.graph = G
        self.dataset_name = 'YelChi'
        train_mask, val_mask, test_mask = split_yelp_data(labels)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def _load(self):
        # print(len(labels))
        full_train_indices = get_full_train_mask(self.train_mask, self.val_mask, self.test_mask, len(self.labels))
        sampled_train_indices = sample_train_mask(full_train_indices, self.args, self.labels, seed=self.args.repeated_runs)
        self.train_mask = _sample_mask(sampled_train_indices, len(self.labels))
        # g = DGLGraph(self.graph)
        # self.g = g
        self.g, degs = handle_nhops(self.graph, self.args, self.args.use_exact_n_hops, self.dataset_name, custom_id=str(self.relations))
        self.num_labels = self.labels.max()+1
        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.g.number_of_nodes()))
        print('  NumEdges: {}'.format(self.g.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.labels.max()))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))
        # assert unsup_sparsemax_weights, 'implement this
        unsup_sparsemax_weights = gat_utils.get_labeling_mask([self.train_mask, self.val_mask, self.test_mask])
        self.unsup_sparsemax_weights = unsup_sparsemax_weights


    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        g = DGLGraph(self.graph)
        g.ndata['train_mask'] = self.train_mask
        g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['label'] = self.labels
        g.ndata['feat'] = self.features
        return g

    def __len__(self):
        return 1

