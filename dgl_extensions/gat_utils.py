from dgl_extensions.gat_models import GAT, SparseGATEdgeFeat, SparseGAT, SoftSparseGAT
from dgl_extensions.gat_models import LearnableSoftSparseGAT
from dgl_extensions.gat_models import UnsupSparseGAT
import torch.nn.functional as F
import torch
# import pickle
# import os
import numpy as np
import random
import scipy.io as sio
import networkx as nx
import os
import pickle

from data_reader.citation_graph import _sample_mask
import pdb
import warnings

# import dgl
# from data_reader.nclass_ppi import CustomPPIDataset
# from torch.utils.data import DataLoader

def build_model(g, args, num_feats, n_classes, heads):

    if args.model.lower() == 'gat':
        gat_model = GAT
    elif args.model.lower() == 'sparsegat':
        gat_model = SparseGAT
    elif args.model.lower() == 'sparsegatedgefeat':
        gat_model = SparseGATEdgeFeat
    elif args.model.lower() == 'softsparsegat':
        gat_model = SoftSparseGAT
    elif args.model.lower() == 'unsupsparsegat':
        gat_model = UnsupSparseGAT
    elif args.model.lower() == 'learnablesoftsparsegat':
        gat_model = LearnableSoftSparseGAT
    else:
        raise NotImplementedError
    att_loss = args.attention_loss
    p_norm = extract_p_norm(att_loss)

    if att_loss.lower() == 'sparse_max':
        softmax_fn_name = att_loss
    else:
        softmax_fn_name = 'edge_softmax'

    model = gat_model(g,
                      args.num_layers,
                      num_feats,
                      args.num_hidden,
                      n_classes,
                      heads,
                      F.elu,
                      args.in_drop,
                      args.attn_drop,
                      args.alpha_sparsemax,
                      args.residual,
                      p_norm=p_norm,
                      softmax=softmax_fn_name,
                      lambda_sparsemax=args.lambda_sparsemax,
                      alpha_sparsemax = args.alpha_sparsemax,
                      unsup_sparsemax_weights = args.unsup_sparsemax_weights,
                      # use_cuda=args.use_cuda,
                      )
    return model


def extract_p_norm(attention_loss):
    att_loss = attention_loss.lower()
    att_loss_split = att_loss.split('l')
    if len(att_loss_split) >= 2:
        p_norm = float(att_loss_split[-1])
    else:
        p_norm = None
    return p_norm
def compute_total_loss(args, logits, labels, loss_fcn, attention_weights_list, lsqr_norm, num_nodes, computed_ce=None):
    if args.model != 'gat':
        # loss = attention_weight
        if computed_ce is None:
            entropy_loss = loss_fcn(logits, labels.float())
        else:
            entropy_loss = computed_ce
        if args.main_loss == 'sparsity_only':
            att_loss = compute_attention_loss(args.attention_loss, attention_weights_list, lsqr_norm, args, num_nodes)
            loss = att_loss * args.attention_weight 
            # print('args.attention_loss {}'.format(att_loss))
        elif args.main_loss == 'entropy_and_sparsity':
            att_loss = compute_attention_loss(args.attention_loss, attention_weights_list, lsqr_norm, args, num_nodes)
            loss = att_loss * args.attention_weight + entropy_loss
            # print('args.attention_loss {}'.format(att_loss))
        elif args.main_loss == 'entropy_only':
            # dont change the loss
            loss = entropy_loss
        elif args.main_loss == 'no_loss':
            loss = entropy_loss * 0
        else:
            print('WARNING_LOSS_MISSING {}'.format(args.main_loss))

            raise NotImplementedError

    compute_sparsity_stats(attention_weights_list)
    return loss

def compute_sparsity_stats(attention_weights_list):
    all_zeros_entries = []
    all_elems = []

    # pdb.set_trace()
    for att in attention_weights_list:
        n_zeroes_entries = (att == 0).sum()
        all_zeros_entries.append(n_zeroes_entries)
        all_elems.append(torch.numel(att))
    total_number_sparse_att = np.sum(np.array(all_zeros_entries))
    total_number_att = np.sum(np.array(all_elems))

    sparsity_percentage = float(total_number_sparse_att) / total_number_att
    print('Sparsity percent: {:.2f} total_number sparse att {}  total_number_att {}'.format(sparsity_percentage*100, total_number_sparse_att, total_number_att))
    return sparsity_percentage


def compute_attention_loss(attention_loss, attention_weights_list, norm_val, args, num_nodes):
    # stack the attention_weights from all layers to large array
    # attention_weigths list have shape 2000x 8x 1 (for first 2 layers) + 1 for last layer.
    # why are they 8x? -> neighbor sampling
    if attention_loss == 'sparse_max':
        # no need to do anything
        return 0
    total_loss = 0
    n_att_layers = len(attention_weights_list)
    for i, (attention_weights, norm) in enumerate(zip(attention_weights_list, norm_val)):
        if i == n_att_layers - 1 and not args.att_last_layer:
            break
        else:
            attention_weights = attention_weights.squeeze()
            if norm_val[0] is not None:
                norm = norm.squeeze()
            if attention_loss.lower() == 'l0':
                loss = torch.norm(attention_weights, p=0, dim=0).mean()
                #print(attention_weights[0])
            elif attention_loss.lower() == 'l1':
                # mean over all attention weights & sum over heads.
                loss = torch.norm(attention_weights, 1, 0).mean()
                #print(attention_weights[0])
            elif attention_loss.lower() == 'l2':
                loss = torch.square(attention_weights).sum(dim=0).mean()
            elif attention_loss.lower() == 'negl2':
                loss = - torch.square(attention_weights).sum(dim=0).mean()

            else: 
                eps = 1e-40
                raw_weights = attention_weights
                attention_weights = raw_weights + eps # avoid nans
                # raw_weights = attention_weights.clone()

                if attention_loss.lower() == 'log':
                    log_loss = - (torch.log(attention_weights)).sum(dim=0).mean()
                    loss = log_loss
                elif attention_loss.lower().startswith('l0.'):
                    # mean over all heads, sum over all samples
                    loss = norm.sum(dim=0).mean()
                elif attention_loss.lower() == 'min_entropy':
                    # assert False, 'Why can it be negative???.'
                    # - sum p *log(p)
                    entropy = - (raw_weights * torch.log(attention_weights)).sum(dim=0).mean()
                    loss = entropy
                else:
                    print('WARNING_LOSS_MISSING {}'.format(attention_loss))
                    raise NotImplementedError
            # Need loss per node
            total_loss += loss
        total_loss /= (num_nodes * n_att_layers)

        # print('attention_loss {}'.format(total_loss))
    return total_loss

def get_train_idx_ppi(labeling_rate, samples_per_class, seed=None):
    # def parse_labeling_ratio(labeling_rate, samples_per_class):
    if samples_per_class == -1:
        labeling_rate = labeling_rate
    else:
        assert False, 'Not yet supported for PPI, not sure how to do it best.'
        samples_per_class = samples_per_class
    train_idx = np.arange(1, 21)
    n_train_graphs = len(train_idx)
    train_n_samples = int(n_train_graphs * labeling_rate)
    assert train_n_samples > 0, 'Too low labeling_rate for PPI {}'.format(labeling_rate)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    random_train_idx = np.random.permutation(train_idx)
    train_idx_random_selected = random_train_idx[:train_n_samples]
    return train_idx_random_selected

def bug_get_full_train_mask(train_mask, val_mask, test_mask, n_samples):
    # get all indices
    train_idx = train_mask.nonzero()[0]
    val_idx = val_mask.nonzero()[0]
    test_idx = test_mask.nonzero()[0]
    all_samples_indx = set(range(0, n_samples))
    warnings.warn('Removing too many samples from training. Not too critical, but needs to redo experiments in the end with these samples included')
    full_train_indices = all_samples_indx - set(train_idx)
    full_train_indices = full_train_indices - set(val_idx)
    full_train_indices = full_train_indices - set(test_idx)
    full_train_indices = list(full_train_indices)
    return full_train_indices


def get_full_train_mask(train_mask, val_mask, test_mask, n_samples):
    # get all indices
    train_idx = train_mask.nonzero()[0]
    val_idx = val_mask.nonzero()[0]
    test_idx = test_mask.nonzero()[0]
    all_samples_indx = set(range(0, n_samples))
    full_train_indices = all_samples_indx 
    full_train_indices = full_train_indices - set(val_idx)
    full_train_indices = full_train_indices - set(test_idx)
    full_train_indices = list(full_train_indices)
    return full_train_indices

def sample_train_mask(train_idx, args, labels, seed=None):
    labeling_rate = args.labeling_rate
    samples_per_class = args.samples_per_class
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    num_classes = len(set(labels))
    if samples_per_class == -1:
        # assert False, 'Switch to samples/class. It is more stable.'
        labeling_rate = labeling_rate
        n_train_samples = len(train_idx)
        train_n_samples = int(n_train_samples * labeling_rate)
        uniform_train_idx = sample_fixed_number_per_class(train_idx, labels, samples_per_class=1)
        assert train_n_samples >= num_classes, 'Too low labeling_rate  {} for {}'.format(labeling_rate, args.data)
        remaining_train_idx = remove_indices(train_idx, uniform_train_idx)

        remaining_train_n_samples = train_n_samples - len(uniform_train_idx)
        random_train_idx = np.random.permutation(remaining_train_idx)
        train_idx_random_selected = random_train_idx[:remaining_train_n_samples]
        train_idx_random_selected = np.concatenate([uniform_train_idx, train_idx_random_selected], axis=0)
        train_labels = labels[train_idx_random_selected] 
        assert len(train_labels) == train_n_samples
        print(labels[train_idx_random_selected])
        assert len(set(train_labels)) == labels.max()+1, 'Each class should have at least one samples'
    else:
        samples_per_class = args.samples_per_class
        train_idx_random_selected = sample_fixed_number_per_class(train_idx, labels, samples_per_class)
    print(train_idx_random_selected)
    return train_idx_random_selected


def sample_fixed_number_per_class(train_idx, labels, samples_per_class):
    '''

    return combined_train_indices with n samples/class
    '''
    train_idx = np.array(train_idx)
    all_class = range(0, labels.max()+1)
    train_labels = labels[train_idx]
    train_idx_random_selected = []

    for class_number in all_class:
        train_samples_from_this_class = train_idx[train_labels == class_number]
        random_ordered_samples = np.random.permutation(train_samples_from_this_class)
        sampled_train_idx = random_ordered_samples[:samples_per_class]

        assert len(sampled_train_idx) == samples_per_class, 'Requesting too many samples {} from max {}'.format(samples_per_class, len(sampled_train_idx))
        train_idx_random_selected.append(sampled_train_idx)


    results = np.concatenate(train_idx_random_selected, axis=0)
    return results


def remove_indices(idx, to_remove_idx):
    remaining_idx = np.array(list(set(list(idx)) - set(list(to_remove_idx))))
    # assert False, 'wrong array'
    return remaining_idx


# def sample_from_class(labels, target_class, nums=1):
#     indices = (labels == target_class).nonzero()
#     random_indices = np.random.permutation(indices)[:nums]

#     return torch.LongTensor(random_indices)


def get_idx_ell(labeling_rate, samples_per_class, seed=None):
    # def parse_labeling_ratio(labeling_rate, samples_per_class):
    if samples_per_class == -1:
        labeling_rate = labeling_rate
    else:
        assert False, 'Not yet supported for PPI, not sure how to do it best.'
        samples_per_class = samples_per_class
    train_idx = np.arange(0, 36)
    test_idx = np.arange(36, 49)

    valid_seed = 42

    valid_idx = sample(train_idx, seed=valid_seed, num=4)
    remaining_idx = torch.LongTensor(list(set(list(train_idx)) - set(list(valid_idx))))

    n_train_graphs = len(remaining_idx)
    train_n_samples = int(n_train_graphs * labeling_rate)
    assert train_n_samples > 0, 'Too low labeling_rate for Elliptic-dataset {}'.format(labeling_rate)
    train_idx_random_selected = sample(remaining_idx, seed=seed, num= train_n_samples)
    return train_idx_random_selected, valid_idx, test_idx


def sample(idx, seed, num):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    random_train_idx = np.random.permutation(idx)
    idx_random = random_train_idx[:num]
    return idx_random


# def build_ppi_dataset(args):
#     dataset = args.data
#     batch_size = args.batch_size
#     if dataset.lower() == 'ppi':
#         valid_dataset = CustomPPIDataset(mode='valid', args=args)
#         test_dataset = CustomPPIDataset(mode='test', args=args)
#         train_dataset = CustomPPIDataset(mode='train', args=args)
#         valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
#     # elif dataset.lower() in ['citeseer', 'cora', 'pubmed']:
#     #     ...
#     else:
#         raise NotImplementedError
    # return train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader

def save_pickle(cache_file, data):
    with open(cache_file, 'wb') as f: pickle.dump(data, f)
def read_pickle(cache_file):
    with open(cache_file, 'rb') as f: data = pickle.load(f)
    return data

def exec_if_not_done(file='', func=None, redo=False):
    if not os.path.exists(file) or redo:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        results = func()
        save_pickle(file, results)
    else:
        results = read_pickle(file)
    return results

def load_yelp(prefix='data/yelp_chi/', file_name='YelpChi.mat', relations=['net_rur', 'net_rtr', 'net_rsr']):

    data = sio.loadmat(prefix + file_name)
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = truelabels.tolist()[0]
    features = features.todense()

    # relations =['net_rtr']
    adj_mat = [data[relation] for relation in relations]

    # assert len(adj_mat[0].nonzero()[0]) <= 100000
    # pdb.set_trace()
    # warnings.warn('leaving out graph parts to test fast')
    # return None, None, features, truelabels

    # N = features.shape[0]
    # gs = [nx.to_networkx_graph(adj) for adj in adj_mat]

    adj_main = np.sum(adj_mat) # change the index to specify which adj matrix to use for aggregation
    adj_main[adj_main > 1] = 1
    # G = nx.to_networkx_graph(adj_main)
    # pdb.set_trace()
    G = exec_if_not_done(file='data/yelp/nx_graph_{}.bin'.format(relations),
                         func=lambda: nx.DiGraph(adj_main),
                         )


    return G, adj_mat, features, truelabels

def split_yelp_to_idx(labels, valid_ratio=.14, test_ratio=.30):
    labels = np.array(labels)
    assert len(labels.shape) == 1, 'need to adapt code if label is 2-dimensional'

    samples_indices = np.arange(len(labels))
    n_samples_total = len(samples_indices)
    n_train_samples = int(n_samples_total * (1 - valid_ratio - test_ratio))
    n_valid_samples = int(n_samples_total * valid_ratio)
    n_test_samples = int(n_samples_total * test_ratio)

    set_seed(seed=42)
    permuted_idx = np.random.permutation(samples_indices)
    train_idx = permuted_idx[:n_train_samples]
    valid_idx = permuted_idx[n_train_samples:n_train_samples + n_valid_samples]
    start = n_train_samples + n_valid_samples
    # end = n_train_samples + n_valid_samples + n_test_samples
    test_idx = permuted_idx[start:]

    assert n_samples_total == len(labels), 'the entire dataset is labeled'

    all_idx = np.concatenate([train_idx, valid_idx, test_idx])
    # assert set(),"samples are not unique"
    # assert set(train_idx) + set(valid_idx) + set(test_idx) == set(labeled_samples_idx), 'missing on samples '
    assert len(train_idx) > 0
    assert len(valid_idx) > 0
    assert len(test_idx) > 0
    assert len(all_idx) == len(permuted_idx)
    assert (all_idx == permuted_idx).sum() == len(all_idx), 'not taking all expamples'
    return train_idx, valid_idx, test_idx





def set_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)

def split_yelp_data(labels):
    '''
    split the data deterministically.
    '''
    train_idx, valid_idx, test_idx = split_yelp_to_idx(labels)
  
    assert abs(len(valid_idx) - len(train_idx) * .25) <= 1
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(labels)

    train_mask = _sample_mask(train_idx, len(labels))
    valid_mask = _sample_mask(valid_idx, len(labels))
    test_mask = _sample_mask(test_idx, len(labels))
    return train_mask, valid_mask, test_mask

def get_labeling_mask(list_of_masks):
    '''
    first mask = train
    returns a torch vector to mark which sample is unlabeled
    unlabeled = 0
    labeled = 1
    valid and test = 1
    '''
    assert len(list_of_masks) == 3
    # assert len(list_of_masks[0]) > len(list_of_masks[1])
    # assert len(list_of_masks[0]) > len(list_of_masks[2])
    # assert False, 'implement'
    combined_mask = list_of_masks[0] + list_of_masks[1] + list_of_masks[2]
    # negate the masks.
    reweighting_vector = combined_mask.astype('double')

    # assert reweighting_vector.sum() + combined_mask.sum() == len(combined_mask), 'missing samples'

    return reweighting_vector