"""Main training file for citation graph datasets."""

import argparse
import numpy as np
import time
from model_archs.gat_utils import build_model
from model_archs.gat_utils import collect_attention_weights
from model_archs.gat_utils import compute_sparsity_stats
from model_archs.gat_utils import compare_tree_sparsification
from model_archs.gat_utils import compute_att_on_same_class
from analysis.attention_vis import plot_attentions
from analysis.attention_vis import plot_att_trees
from analysis.attention_analysis import plot_attention_entropy
from analysis.attention_analysis import plot_unlabeled_samples
from analysis.attention_analysis import analyze_noise_detection
from analysis.attention_analysis import compute_att_on_shortest_paths
from data_reader.citation_graph_dgl import load_data
from sklearn.metrics import f1_score
from sklearn import metrics
import copy
import torch


def accuracy(logits, labels):
  """Compute an evaluation score based on logits and labels.

  Args:
      logits (tensor): Output logits.
      labels (tensor): Labels of all nodes.

  Returns:
      score: evaluation score.
  """
  _, indices = torch.max(logits, dim=1)
  correct = torch.sum(indices == labels)
  return correct.item() * 1.0 / len(labels)


def compute_f1_score(logits, labels, average_mode='micro'):
  """Compute an evaluation score based on logits and labels.

  Args:
      logits (tensor): Output logits.
      labels (tensor): Labels of all nodes.
      average_mode (str, optional): Description

  Returns:
      score: evaluation score.
  """
  _, indices = torch.max(logits, dim=1)

  score = f1_score(
      labels.data.cpu().numpy(), indices.cpu().numpy(), average=average_mode)
  return score


def compute_f1_score_binary(logits, labels):
  """Compute an evaluation score based on logits and labels.

  Args:
      logits (tensor): Output logits.
      labels (tensor): Labels of all nodes.

  Returns:
      score: evaluation score.
  """
  probs = torch.softmax(logits, dim=1)[:, 1]
  probs[probs >= 0.5] = 1
  probs[probs < 0.5] = 0

  score = f1_score(
      labels.data.cpu().numpy(), probs.cpu().numpy(), average=None)[1]
  return score


def evaluate(model, features, labels, mask, args):
  """Evaluate a trained model.

  Args:
      model (model): graph model.
      features (tensor): input features of all nodes.
      labels (tensor): labels of all nodes.
      mask (tensor): mask to choose samples
      args (namespace): all hyperparameters.

  Returns:
      score: evaluation score.
      secondary_score: a second score.
  """
  model.eval()
  with torch.no_grad():
    logits = model(features)
    if args.data.startswith('yelp'):
      logits = logits[mask]
      labels = labels[mask]
      average_mode = 'binary'
      probs = torch.softmax(logits, dim=1)[:, 1]
      fpr, tpr, _ = metrics.roc_curve(labels.data.cpu().numpy(),
                                      probs.cpu().numpy())
      auc_scores = metrics.auc(fpr, tpr)
      score = auc_scores
      f1_binary = compute_f1_score_binary(logits, labels)

      secondary_score = f1_binary
      score = auc_scores
    else:
      logits = logits[mask]
      labels = labels[mask]
      average_mode = 'micro'
      secondary_score = compute_f1_score(logits, labels, average_mode)
      score = accuracy(logits, labels)

    return score, secondary_score


def main(args):
  """Training and computation of some stats.

  This function handles the entire training process and analyses the learned
  results.

  Args:
      args (namespace): All hyperparameters

  Returns:
      resutls: Collected results after traning.
  """
  if args.model == 'sparselossgat' and args.attention_weight == 0:
    args.model = 'unsupsparsegat'
  assert args.labeling_rate is None
  assert (args.samples_per_class is None or
          args.samples_per_class > -1) or args.use_provided_data_split
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  data = load_data(args)
  features = torch.FloatTensor(data.features)
  labels = torch.LongTensor(data.labels)

  if hasattr(torch, 'BoolTensor'):
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    trainable_masks = torch.ones(len(train_mask), dtype=torch.bool)
    trainable_masks[val_mask] = False
    trainable_masks[test_mask] = False
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
        (n_edges, n_classes, train_mask.int().sum().item(),
         val_mask.int().sum().item(), test_mask.int().sum().item()))

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
    trainable_masks = trainable_masks.cuda()
    if 'w' in data.g.edata:
      data.g.edata['w'] = data.g.edata['w'].cuda()

  g = data.g
  if cuda:
    g = g.to(args.gpu)
  n_edges = g.number_of_edges()
  # create model
  heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
  model = build_model(g, args, num_feats, n_classes, heads)
  print(model)

  if cuda:
    model.cuda()

  loss_fcn = torch.nn.CrossEntropyLoss()
  # use optimizers
  optimizer = torch.optim.Adam(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  # initialize graph
  dur = []
  best_valid_acc = -1
  best_valid_f1 = -1
  best_epoch = -1
  for epoch in range(args.epochs):
    model.train()
    if epoch >= 3:
      t0 = time.time()
    # forward
    logits = model(features)
    loss = loss_fcn(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if epoch >= 3:
      dur.append(time.time() - t0)

    train_acc = accuracy(logits[train_mask], labels[train_mask])
    train_f1 = compute_f1_score(logits[train_mask], labels[train_mask])

    val_acc, val_f1 = evaluate(model, features, labels, val_mask, args)
    attention_list = collect_attention_weights(
        model,
        mean=True,
        keep_list=True,
    )
    compute_sparsity_stats(attention_list)

    if val_acc >= best_valid_acc:
      best_valid_acc = val_acc
      best_valid_f1 = val_f1
      best_test_acc, best_test_f1 = evaluate(model, features, labels, test_mask,
                                             args)
      best_epoch = epoch
      best_model = copy.deepcopy(model.state_dict())
    if epoch - best_epoch >= args.patience:
      break

    print('Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |'
          ' ValAcc {:.4f} | Testacc {:.4f} |'
          ' TrainF1 {:.4f} | ValF1 {:.4f} |'
          ' TestF1 {:.4f}| ETputs(KTEPS) {:.2f}'.format(
              epoch, np.mean(dur), loss.item(), train_acc, val_acc,
              best_test_acc, train_f1, val_f1, best_test_f1,
              n_edges / np.mean(dur) / 1000))

  print()
  print('Test Accuracy {:.4f} Test F1 (pos-class) {:.4f}'.format(
      best_test_acc, best_test_f1))
  more_info = (best_epoch, best_test_acc, best_test_f1, best_valid_acc,
               best_valid_f1)
  model.load_state_dict(best_model)
  best_test_acc, best_test_f1 = evaluate(model, features, labels, test_mask,
                                         args)
  attention_weights = collect_attention_weights(model, mean=True)
  more_info = [best_epoch, best_test_acc, best_valid_acc, None]

  attention_list = collect_attention_weights(
      model,
      mean=True,
      keep_list=True,
  )
  if args.compute_tree_sparsification:
    saving, average_saving, most_sparse_node = compare_tree_sparsification(
        g, attention_list)
    print('Saving %f Best Node: %s sparse_rate: %f' % (
        average_saving,
        most_sparse_node,
        saving[most_sparse_node],
    ))
    plot_att_trees(
        g,
        attention_list,
        labels,
        model_name=args.model,
        data=args.data,
        samples_per_class=args.samples_per_class,
        best_epoch=best_epoch,
        nodes=[most_sparse_node],
    )
  if args.plot_tree:
    attention_list = attention_list[:3]
    compute_sparsity_stats(attention_list)
    plot_att_trees(
        g,
        attention_list,
        labels,
        model_name=args.model,
        data=args.data,
        samples_per_class=args.samples_per_class,
        best_epoch=best_epoch,
    )
  if args.noise_injected_ratio > 0:
    noisy_edges_ids = data.noisy_edges_ids
    n_nodes = len(g.nodes())
    att_on_noisy_edge = analyze_noise_detection(attention_weights,
                                                noisy_edges_ids, n_nodes)
    more_info[-1] = att_on_noisy_edge
  if args.noise_robustness:
    att_on_same_class = compute_att_on_same_class(g, model, features, labels)
    more_info[-1] = att_on_same_class

  if args.vis_att:
    plot_attentions(g, model, labels, args)

  if args.att_on_shortest_paths:
    att_mean = compute_att_on_shortest_paths(g, attention_weights, labels)
    more_info[-1] = att_mean
  if args.att_analysis:
    attention_heads = collect_attention_weights(model)
    plot_attention_entropy(
        graph=g,
        attention_heads=attention_heads,
        dataset=args.data,
        samples_per_class=args.samples_per_class,
        model_name=args.model,
        labels=labels,
    )
    plot_unlabeled_samples(
        graph=g,
        attention_weights=attention_heads[:, -1],
        train_mask=train_mask,
        dataset=args.data,
        samples_per_class=args.samples_per_class,
        model_name=args.model,
        labels=labels,
    )

  assert len(more_info) == 4, 'Too many elements in the list %d' \
                              % len(more_info)

  return best_test_acc, more_info


def get_args():
  """Provides a parser with standard arguments.

  Returns:
      parser: parser with standard arguments
  """
  parser = argparse.ArgumentParser(description='GAT')
  parser.add_argument(
      '--gpu', type=int, default=1, help='which GPU to use. Set -1 to use CPU.')
  parser.add_argument(
      '--epochs', type=int, default=400, help='number of training epochs')
  parser.add_argument(
      '--num-heads',
      type=int,
      default=8,
      help='number of hidden attention heads')
  parser.add_argument(
      '--num-out-heads',
      type=int,
      default=1,
      help='number of output attention heads')
  parser.add_argument(
      '--num-layers', type=int, default=2, help='number of hidden layers')
  parser.add_argument(
      '--num-hidden', type=int, default=8, help='number of hidden units')
  parser.add_argument(
      '--residual',
      action='store_true',
      default=False,
      help='use residual connection')
  parser.add_argument(
      '--in-drop', type=float, default=.6, help='input feature dropout')
  parser.add_argument(
      '--attn-drop', type=float, default=.6, help='attention dropout')
  parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
  parser.add_argument(
      '--weight-decay', type=float, default=5e-4, help='weight decay')
  parser.add_argument(
      '--alpha',
      type=float,
      default=0.2,
      help='the negative slop of leaky relu')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=2,
      help='batch size used for training, validation and test')
  parser.add_argument(
      '--patience', type=int, default=20, help='used for early stop')
  parser.add_argument(
      '--dropout', type=float, default=0, help='Dropout for non-gat models ')
  parser.add_argument(
      '--graphsage_aggregator_type',
      type=str,
      default='gcn',
      help='Aggregator type: mean/gcn/pool/lstm')

  parser.add_argument(
      '--data', type=str, default=10, help='cora | citeseer | pubmed')

  parser.add_argument('--model', type=str, help='which model to be used')
  parser.add_argument(
      '--labeling_rate',
      type=float,
      default=None,
      help='how much labels should be used?')
  parser.add_argument(
      '--samples_per_class',
      type=float,
      default=-1,
      help='Alternative: how mnay samples per classes should be used. -1 means use labeling rate instead'
  )
  parser.add_argument('--attention_weight', type=float, default=1)
  parser.add_argument('--lambda_sparsemax', type=float, default=0)
  parser.add_argument(
      '--alpha_sparsemax',
      type=float,
      default=0,
      help='1 = only sparse_max, 0 = only softmax')

  parser.add_argument('--layer_aggregation', type=bool)
  parser.add_argument('--self_loop', type=bool)

  parser.add_argument('--correct_data_split', type=bool, default=True)
  parser.add_argument('--label_prop_steps', type=float, default=None)
  parser.add_argument('--pseudo_label_mode', type=str, default='final_only')
  parser.add_argument('--use_adj_matrix', type=bool, default=None)
  parser.add_argument(
      '--vis_att',
      type=bool,
      default=None,
      help='visualize attention after training ')
  parser.add_argument(
      '--n_nodes_to_plot', type=int, default=20, help='How many nodes to plot.')
  parser.add_argument(
      '--noise_robustness',
      type=bool,
      default=None,
      help='Performs noise robustness experiments or not')
  parser.add_argument(
      '--noise_injected_ratio',
      type=float,
      default=0,
      help='Percentual noise to be added.')
  parser.add_argument(
      '--noise_type', type=str, default='random', help='random | heterophily')
  parser.add_argument(
      '--att_on_shortest_paths',
      type=bool,
      default=False,
      help='Compute the att-mean on shortest path')
  parser.add_argument(
      '--plot_tree',
      type=bool,
      default=False,
      help='Plot explanation trees for some nodes')
  parser.add_argument(
      '--compute_tree_sparsification',
      type=bool,
      default=False,
      help='How much of the interpretability tree is saved')
  parser.add_argument(
      '--att_analysis',
      type=bool,
      default=False,
      help='Plot explanation trees for some nodes')

  return parser


if __name__ == '__main__':
  parser = get_args()
  args = parser.parse_args()

  config = args.__dict__
  config['data'] = 'cora'
  config['gpu'] = -1
  config['attn_drop'] = 0
  config['in_drop'] = 0
  config['SLGAT_apply_on_up_to_layers'] = 3
  config['test_level'] = 'ALL'
  config['patience'] = 100
  config['num_epochs'] = 1000
  config['SLGAT_loss'] = 'L2'
  config['SLGAT_apply_on_all_samples'] = True
  config['SLGAT_use_same_projection'] = False
  config['SLGAT_projection_detach'] = False
  config['model'] = 'sparselossgat'
  config['labeling_rate'] = -2

  print(args)
  main(args)
