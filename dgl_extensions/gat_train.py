"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import argparse
import numpy as np
import networkx as nx
import time
import torch
# import torch.nn.functional as F
from dgl import DGLGraph
# from dgl.data import load_data
# from gat import GAT
# from utils import EarlyStopping

# import numpy as np
# import torch
# import dgl
import argparse
# from sklearn.metrics import f1_score
# from dgl.data.ppi import LegacyPPIDataset
from dgl_extensions.gat_utils import build_model
from dgl_extensions.gat_utils import compute_total_loss
from data_reader.citation_graph_dgl import load_data
import pdb
from sklearn.metrics import f1_score

# from dgl_extensions.gat_utils import build_dataset

# import dgl
# from data_reader.nclass_ppi import CustomPPIDataset
# from torch.utils.data import DataLoader
# import utils as u
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def compute_f1_score(logits, labels, average_mode='micro'):
    # predict = np.where(logits.data.cpu().numpy() >= 0., 1, 0)
    _, indices = torch.max(logits, dim=1)

    score = f1_score(labels.data.cpu().numpy(),
                     indices.cpu().numpy(), average=average_mode)
    return score

def evaluate(model, features, labels, mask, args):
    model.eval()
    with torch.no_grad():
        if args.model.lower() != 'gat':
            logits, attention_weights_list, lsqr_norm = model(features)
        else:
            logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        if args.data.startswith('yelp'):
            average_mode = 'binary'
        else:
            average_mode = 'micro'

        return accuracy(logits, labels), compute_f1_score(logits, labels, average_mode)


def main(args):
    # load and preprocess dataset
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = load_data(args)
    args.unsup_sparsemax_weights = data.unsup_sparsemax_weights
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d 
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        if 'w' in data.g.edata:
            data.g.edata['w'] = data.g.edata['w'].cuda()
        args.unsup_sparsemax_weights = args.unsup_sparsemax_weights.cuda()


    # g = data.graph
    # # add self loop
    # g.remove_edges_from(nx.selfloop_edges(g))
    # g = DGLGraph(g)
    # g.add_edges(g.nodes(), g.nodes())

    g = data.g
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = build_model(g, args, num_feats, n_classes, heads)
    print(model)
    # if args.early_stop:
    # stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    best_valid_score = -1
    best_epoch = -1
    best_test_perf = -1
    # last_best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        if 'gat' == args.model:
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
        else:
            logits, attention_weights_list, lsqr_norm = model(features)
            ce_loss = loss_fcn(logits[train_mask], labels[train_mask])
            loss = compute_total_loss(args, logits, labels, loss_fcn,
                                      attention_weights_list, lsqr_norm,
                                      model.g.number_of_nodes(),
                                      computed_ce=ce_loss)

        optimizer.zero_grad()
        # pdb.set_trace()
        loss.backward()
        # pdb.set_trace()

        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        train_f1 = compute_f1_score(logits[train_mask], labels[train_mask])

        # if args.fastmode:
        #     val_acc = accuracy(logits[val_mask], labels[val_mask])
        # else:
        val_acc, val_f1 = evaluate(model, features, labels, val_mask, args)

        # pdb.set_trace()
        if args.data.lower().startswith('yelp'):
            val_score = val_f1
        else:
            val_score = val_acc
        if val_score > best_valid_score:
            # last_best_epoch = best_epoch
            best_valid_score = val_score
            best_test_acc, best_test_f1 = evaluate(model, features, labels, test_mask, args)
            best_epoch = epoch
        if epoch - best_epoch >= args.patience:
            break


        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | Testacc {:.4f} |"
              " TrainF1 {:.4f} | ValF1 {:.4f} |"
              " TestF1 {:.4f}| ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, best_test_acc, train_f1, val_f1, best_test_f1, n_edges / np.mean(dur) / 1000))

    print()
    # if args.early_stop:
    # model.load_state_dict(torch.load('es_checkpoint.pt'))
    print("Test Accuracy {:.4f} Test F1 (pos-class) {:.4f}".format(best_test_acc, best_test_f1))
    more_info = (best_epoch, best_test_acc, best_test_f1, best_valid_score, val_f1)
    return best_test_acc, more_info


def get_args():
    parser = argparse.ArgumentParser(description='GAT')
    # parser = u.Namespace({})
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=20,
                        help="used for early stop")    
    # parser.add_argument("--fastmode", action="store_true", default=False,
                        # help="fast_evaluation")
    # parser.add_argument('--early-stop', action='store_true', default=True,
                        # help="indicates whether to use early stop or not")

    parser.add_argument('--data', type=str, default=10,
                        help="cora | citeseer | pubmed")

    parser.add_argument('--model', type=str, default=10,
                        help="GAT | SparseGAT | SparseGATEdgeFeat")
    parser.add_argument('--labeling_rate', type=float, default=1, help='how much labels should be used?')
    parser.add_argument('--samples_per_class', type=float, default=-1, help= 'Alternative: how mnay samples per classes should be used. -1 means use labeling rate instead')
    parser.add_argument('--attention_weight', type=float, default=1)
    parser.add_argument('--main_loss', type=str, default='entropy_only')
    parser.add_argument('--attention_loss', type=str, default='min_entropy')
    parser.add_argument('--n_hops_max', type=int, default=1)
    parser.add_argument('--use_exact_n_hops', type=bool, default=True)
    parser.add_argument('--att_last_layer', type=bool, default=True)
    # parser.add_argument('--att_last_layer', type=bool, default=True)
    parser.add_argument('--lambda_sparsemax', type=float, default=0)
    parser.add_argument('--alpha_sparsemax', type=float, default=0, help='1 = only sparse_max, 0 = only softmax')


    # ignore cli arguments, loading default params only
    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)

