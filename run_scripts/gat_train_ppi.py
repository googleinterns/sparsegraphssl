"""Graph Attention Networks (PPI Dataset) in DGL"""

import numpy as np
import torch
import dgl
from sklearn.metrics import f1_score
from model_archs.gat_utils import build_model
import argparse

from data_reader.nclass_ppi import CustomPPIDataset
from data_reader.nclass_ppi import SubsampledPPIDataset
from data_reader.nclass_ppi import UnsupSubgraphPPIDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate(sample):
  """Collects the sampled data-points from the batch.

  This function concatenates batch-data together into one graph.

  Args:
      sample (tuple): samples from the data-loader

  Returns:
      resutls: tuple of results.
  """
  data = tuple(map(list, zip(*sample)))
  using_ppi_subgraph_ssl = len(list(data)) == 4
  using_ppi_subsampled = len(list(data)) == 5
  graphs, feats, labels = data[:3]

  graph = dgl.batch(graphs)
  feats = torch.from_numpy(np.concatenate(feats))
  labels = torch.from_numpy(np.concatenate(labels))

  results = [graph, feats, labels]
  if using_ppi_subgraph_ssl or using_ppi_subsampled:
    labeled_samples = data[3]
    labeled_samples_vector = torch.from_numpy(np.concatenate(labeled_samples))
    results.append(labeled_samples_vector)
    # if using_ppi_subsampled:
    unsup_sparsemax_weights = data[4]
    unsup_spm_weights = torch.from_numpy(
        np.concatenate(unsup_sparsemax_weights))
    results.append(unsup_spm_weights)

  return tuple(results)


def evaluate(feats, model, subgraph, labels, loss_fcn, args):
  """Evaluate a trained model.

  Args:
      feats (tensor): input features.
      model (model): graph model.
      subgraph (graph): subgraph to use for computation.
      labels (tensor): labels of all nodes.
      loss_fcn (loss-fucntion):  which loss fucntion to be used.
      args (namespace): all hyperparameters.

  Returns:
      score: evaluation score.
      secondary_score: a second score.

      mask (tensor): mask to choose samples
  """
  model
  with torch.no_grad():
    model.eval()

    model.g = subgraph
    for layer in model.graph_layers:
      layer.g = subgraph

    output = model(feats.float())

    if -1 in labels:
      labeled_samples = labels.nonzero()
      output = labeled_samples[labeled_samples]
      labels = labeled_samples[labeled_samples]

    loss_data = loss_fcn(output, labels.float())
    predict = np.where(output.data.cpu().numpy() >= 0., 1, 0)

    mode = 'micro'

    score = f1_score(labels.data.cpu().numpy(), predict, average=mode)
    return score, loss_data.item()


def compute_test_score(test_dataloader, loss_fcn, model, device, args):
  """Computes test_score.

  Args:
      test_dataloader (data-loader): the data-loader of test.
      loss_fcn (loss-fucntion):  which loss fucntion to be used.
      model (model): graph model.
      device (str): gpu.
      args (namespace): all hyperparameters.

  Returns:
      TYPE: Description
  """
  test_score_list = []
  n_samples = []
  for batch, test_data in enumerate(test_dataloader):
    subgraph, feats, labels = test_data
    assert len(feats) > 0, 'empty data'
    if args.use_cuda:
      feats = feats.cuda()
      labels = labels.cuda()
      if 'w' in subgraph.edata:
        subgraph.edata['w'] = subgraph.edata['w'].cuda()
      subgraph = subgraph.to(args.gpu)
    test_score = evaluate(feats, model, subgraph, labels.float(), loss_fcn,
                          args)[0]
    test_score_list.append(test_score * len(feats))
    n_samples.append(len(labels))

  best_test_perf = np.array(test_score_list).sum() / np.sum(n_samples)
  print('Test F1-Score: {:.4f}'.format(best_test_perf))
  return best_test_perf


def set_att_layers(all_layers=[], att_name=None, att=None, use_cuda=False):
  """Summary

  Args:
      all_layers (list, optional): All atetntion list.
      att_name (None, optional): Name of attribute to be set.
      att (None, optional): Value of attribute to be set.
      use_cuda (bool, optional): Use cuda or not.
  """
  for layer in all_layers:
    if att is not None:
      setattr(layer, att_name, att)


def main(args):
  """Training and computation of some stats.

  This function handles the entire training process and analyses the learned
  results.

  Args:
      args (namespace): All hyperparameters

  Returns:
      resutls: Collected results after traning.
  """
  assert args.attn_drop == 0
  assert args.in_drop == 0
  if args.model == 'sparselossgat' and args.attention_weight == 0:
    args.model = 'SparseGAT'
    args.data = 'ppi'
  if args.model == 'labelpropgat':
    assert False, ('the softmax computation in creating pseudolabels is not '
                   'correct.')

  if args.gpu < 0:
    device = torch.device('cpu')
    args.use_cuda = False
  else:
    # pdb.set_trace()
    device = torch.device('cuda:' + str(args.gpu))
    args.use_cuda = True
    torch.cuda.set_device(args.gpu)

  batch_size = args.batch_size
  cur_step = 0
  patience = args.patience
  best_valid_score = -1
  best_test_perf = -1
  best_loss = 10000
  # define loss function
  if args.use_lp_logits:
    loss_fcn = torch.nn.BCEWithLogitsLoss()
  else:
    loss_fcn = torch.nn.BCELoss()

  # create the dataset
  if args.data.lower() == 'elliptic':
    assert False, ('Using wrong labels. Evalutation and Training are taking the'
                   ' wrong labels for learning.')
    test_dataset = EllDataset(mode='test', args=args)
    train_dataset = EllDataset(mode='train', args=args)
    valid_dataset = EllDataset(mode='valid', args=args)
  elif args.data.lower() == 'ppi':
    train_dataset = CustomPPIDataset(mode='train', args=args)
    valid_dataset = CustomPPIDataset(mode='valid', args=args)
    test_dataset = CustomPPIDataset(mode='test', args=args)
  elif args.data.lower() == 'ppi_subsampled':
    train_dataset = SubsampledPPIDataset(mode='train', args=args)
    valid_dataset = CustomPPIDataset(mode='valid', args=args)
    test_dataset = CustomPPIDataset(mode='test', args=args)
  elif args.data.lower() == 'ppi_subgraph_ssl':
    assert args.model == 'sparselossgat', ('no other model is supported for '
                                           'unsupervised learning at the '
                                           'moment.')
    train_dataset = UnsupSubgraphPPIDataset(mode='train', args=args)
    valid_dataset = CustomPPIDataset(mode='valid', args=args)
    test_dataset = CustomPPIDataset(mode='test', args=args)
  test_dataloader = DataLoader(
      test_dataset, batch_size=batch_size, collate_fn=collate)
  valid_dataloader = DataLoader(
      valid_dataset, batch_size=batch_size, collate_fn=collate)
  train_dataloader = DataLoader(
      train_dataset, batch_size=batch_size, collate_fn=collate)
  assert len(test_dataloader) > 0
  assert len(train_dataloader) > 0
  assert len(valid_dataloader) > 0
  # train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader = build_dataset(args)
  n_classes = train_dataset.labels.shape[1]
  num_feats = train_dataset.features.shape[1]
  g = train_dataset.graph
  heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
  # define the model
  model = build_model(g, args, num_feats, n_classes, heads)
  print(model)

  optimizer = torch.optim.Adam(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  if args.use_cuda:
    model = model.cuda()

  best_epoch = -1
  for epoch in tqdm(range(args.epochs)):
    model.train()
    loss_list = []
    for batch, data in enumerate(train_dataloader):
      if len(data) == 3:
        subgraph, feats, labels = data
        labeled_samples = None
      else:
        subgraph, feats, labels, labeled_samples, unsup_sparsemax_weights = data
      if args.use_cuda:
        feats = feats.cuda()
        labels = labels.cuda()
        if 'w' in subgraph.edata:
          subgraph.edata['w'] = subgraph.edata['w'].cuda()
        subgraph = subgraph.to(args.gpu)
      model.g = subgraph
      for layer in model.graph_layers:
        layer.g = subgraph
      if args.data in ['ppi_subsampled', 'ppi_subgraph_ssl']:
        if args.use_cuda:
          labeled_samples = labeled_samples.cuda()
        set_att_layers(
            all_layers=model.graph_layers,
            att_name='custom_unsup_sparsemax_weights',
            att=unsup_sparsemax_weights,
            use_cuda=args.use_cuda)

      logits = model(feats.float())
      if args.data in ['ppi_subsampled', 'ppi_subgraph_ssl']:
        logits = logits[labeled_samples]
        labels = labels[labeled_samples]

      loss = loss_fcn(logits, labels.float())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_list.append(loss.item())
    # pdb.set_trace()
    loss_data = np.array(loss_list).mean()
    print('Epoch {:05d} | Loss: {:.4f}'.format(epoch + 1, loss_data))
    if np.isnan(loss_data):
      # gradients explosion
      best_valid_score = loss_data
      best_test_perf = loss_data
      break
    if args.data in ['ppi_subsampled', 'ppi_subgraph_ssl']:
      set_att_layers(
          all_layers=model.graph_layers,
          att_name='custom_unsup_sparsemax_weights',
          att=None,
          use_cuda=args.use_cuda)
    if epoch % 1 == 0:
      score_list = []
      val_loss_list = []
      for batch, valid_data in enumerate(valid_dataloader):
        subgraph, feats, labels = valid_data
        if args.use_cuda:

          feats = feats.cuda()
          labels = labels.cuda()
          if 'w' in subgraph.edata:
            subgraph.edata['w'] = subgraph.edata['w'].cuda()
          subgraph = subgraph.to(args.gpu)
        score, val_loss = evaluate(feats.float(), model, subgraph,
                                   labels.float(), loss_fcn, args)
        score_list.append(score)
        val_loss_list.append(val_loss)
      mean_score = np.array(score_list).mean()
      mean_val_loss = np.array(val_loss_list).mean()
      print('Val F1-Score: {:.4f} '.format(mean_score))
      # early stop
      if mean_score >= best_valid_score:
        best_valid_score = np.max((mean_score, best_valid_score))
        best_loss = np.min((best_loss, mean_val_loss))
        cur_step = 0
        best_epoch = epoch
        best_test_perf = compute_test_score(test_dataloader, loss_fcn, model,
                                            device, args)
      else:
        cur_step += 1
        if cur_step == patience:
          break

  more_info = (best_epoch, best_test_perf, None, best_valid_score, None)
  return best_test_perf, more_info


def get_args():
  """Provides a parser with standard arguments.

  Returns:
      parser: parser with standard arguments
  """
  parser = argparse.ArgumentParser(description='GAT')
  parser.add_argument(
      '--gpu',
      type=int,
      default=-1,
      help='which GPU to use. Set -1 to use CPU.')
  parser.add_argument(
      '--epochs', type=int, default=600, help='number of training epochs')
  parser.add_argument(
      '--num-heads',
      type=int,
      default=4,
      help='number of hidden attention heads')
  parser.add_argument(
      '--num-out-heads',
      type=int,
      default=6,
      help='number of output attention heads')
  parser.add_argument(
      '--num-layers', type=int, default=2, help='number of hidden layers')
  parser.add_argument(
      '--num-hidden', type=int, default=256, help='number of hidden units')
  parser.add_argument(
      '--residual',
      action='store_true',
      default=True,
      help='use residual connection')
  parser.add_argument(
      '--in-drop', type=float, default=0, help='input feature dropout')
  parser.add_argument(
      '--attn-drop', type=float, default=0, help='attention dropout')
  parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
  parser.add_argument(
      '--weight-decay', type=float, default=0, help='weight decay')
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
      '--model',
      type=str,
      default=10,
      help='GAT | SparseGAT | SparseGATEdgeFeat')
  parser.add_argument(
      '--labeling_rate',
      type=float,
      default=1,
      help='how much labels should be used?')
  parser.add_argument(
      '--samples_per_class',
      type=float,
      default=-1,
      help='Alternative: how mnay samples per classes should be used. -1 means use labeling rate instead'
  )

  parser.add_argument('--attention_weight', type=float, default=1)
  parser.add_argument('--main_loss', type=str, default='entropy_only')
  parser.add_argument('--attention_loss', type=str, default='min_entropy')
  parser.add_argument('--n_hops_max', type=int, default=1)
  parser.add_argument('--use_exact_n_hops', type=bool, default=True)
  parser.add_argument('--att_last_layer', type=bool, default=True)
  parser.add_argument('--lambda_sparsemax', type=float, default=0)
  parser.add_argument(
      '--alpha_sparsemax',
      type=float,
      default=0,
      help='1 = only sparse_max, 0 = only softmax')

  parser.add_argument('--lambda_scope', type=str)
  parser.add_argument('--lambda_learning_form', type=str)
  parser.add_argument('--unsupervised', type=bool)
  parser.add_argument('--layer_aggregation', type=bool)
  parser.add_argument('--self_loop', type=bool)
  parser.add_argument(
      '--jknet_aggregator_type',
      type=str,
      default='sum',
      help='Aggregator type: sum/mean/max')

  parser.add_argument('--lambda_binding', type=bool, default=False)
  parser.add_argument('--lambda_binding_weight', type=float, default=0)
  parser.add_argument('--correct_data_split', type=bool, default=True)
  parser.add_argument('--use_provided_data_split', type=bool, default=True)

  parser.add_argument(
      '--max_pooling',
      type=bool,
      default=False,
      help='attention pooling for gat')
  parser.add_argument(
      '--test_level', type=str, default='Disabled', help='ALL| Test_Data')
  parser.add_argument(
      '--graphsage_aggregator_type',
      type=str,
      default='gcn',
      help='Aggregator type: mean/gcn/pool/lstm')
  parser.add_argument(
      '--unsup_sparsemax_weights',
      type=None,
      default=None,
  )
  parser.add_argument('--dropout', type=float, default=0)
  parser.add_argument('--SLGAT_loss', type=str, default=None)
  parser.add_argument('--SLGAT_weight', type=float, default=None)
  parser.add_argument('--SLGAT_apply_on_up_to_layers', type=int, default=None)
  parser.add_argument('--SLGAT_use_same_projection', type=bool, default=None)
  parser.add_argument('--SLGAT_apply_on_all_samples', type=bool, default=None)

  parser.add_argument('--log_name', type=str, default=None)
  parser.add_argument('--label_prop_steps', type=float, default=None)
  parser.add_argument('--pseudo_label_mode', type=str, default='final_only')
  parser.add_argument('--prop_at_inference', type=bool, default=True)
  parser.add_argument('--prop_at_training', type=bool, default=True)
  parser.add_argument('--temp', type=float, default=1)
  parser.add_argument('--use_lp_logits', type=bool, default=True)

  return parser


if __name__ == '__main__':
  parser = get_args()
  args = parser.parse_args([])
  print(args)
  config = args.__dict__
  config['num_layers'] = 2
  config['labeling_rate'] = .1
  config['samples_per_class'] = 2
  # config['data'] = 'ppi_subsampled'
  config['data'] = 'ppi_subgraph_ssl'
  config['num_hidden'] = 8
  config['unsupervised'] = True
  config['lambda_sparsemax'] = 100
  config['repeated_runs'] = 1
  config['model'] = 'sparselossgat'

  config['SLGAT_weight'] = 0.01
  config['SLGAT_apply_on_all_samples'] = True
  config['test_level'] = 'ALL'

  main(args)
