"""Defines low-level operations to build up larger blocks of GNN-layers."""

# Lint as  python3

import torch


def att_sqrt(p_norm, edges):
  """Compute the p_norm of attention values.

  Compute different L_p norms on the incoming signals over the edges.

  Args:
    p_norm: can an arbitrary non-negative integer.
    edges: container with all data about the edges

  Returns:
    results: a dict with the value saved to sqrt_alpha
  """
  results = {'sqrt_alpha': torch.pow(edges.data['a'], p_norm)}
  return results


def reduce_sum_sqr(p_norm, nodes):
  """Reduces the incoming messages on the nodes by applying sum and 1/pnorm.

  This function performs squared operation with subsequent summation.

  Args:
    p_norm: can an arbitrary non-negative integer.
    nodes: container with all data about the nodes

  Returns:
    results: a dict with the value saved to sum_sqrt_alpha
  """
  results = {
      'sum_sqrt_alpha':
          torch.pow(torch.sum(nodes.mailbox['sqrt_alpha'], dim=1), 1 / p_norm)
  }
  return results


def e_mul_v(e_name, n_name, out_name):
  """Signal multiplication.

  Multiplies incoming signals on edges (e) with signals on target nodes (v)

  Args:
      e_name: info to be used from edge container
      n_name: info to be used from node container
      out_name: name of the variable to save the results

  Returns:
    exec_op: saved results into dict under out_name
  """

  def exec_op(edges):
    """Function wrapper of signal multiplication.

    creates a function to be applied later during message passing.

    Args:
      edges: container with all data about the edges

    Returns:
      results: dict with results.
    """
    input_feat = edges.data[e_name]
    weight = edges.dst[n_name]
    output = input_feat * weight
    results = {out_name: output}
    return results

  return exec_op


def copy_v(n_name, out_name):
  """Copy attributes from target nodes.

  Copy attribute n_name from target nodes.

  Args:
    n_name: attribute name on nodes
    out_name: attribute name used for saving

  Returns:
    exec_op: operation to be executed is message-passing
  """

  def exec_op(edges):
    """Function warpper.

    creates a function to be applied later during message passing.

    Args:
      edges: container of information on the edges

    Returns:
      results: updated dictionary with the results
    """

    weight = edges.dst[n_name]
    results = {out_name: weight}
    return results

  return exec_op


def e_sub_v(e_name, n_name, out_name):
  """Substraction of (e)dge and target node v.

  choose the signals on edges, substract signals from nodes.

  Args:
    e_name: attribute name on edge
    n_name: attribute name on node
    out_name: where to save the results

  Returns:
    exec_op: operation to be executed is message-passing
  """

  def exec_op(edges):
    """Function warpper.

    creates a function to be applied later during message passing.

    Args:
      edges: container of information on the edges

    Returns:
      results: updated dictionary with the results
    """
    input_feat = edges.data[e_name]
    weight = edges.dst[n_name]
    output = input_feat - weight
    results = {out_name: output}
    return results

  return exec_op


def u_diff_v_l2(u_name, v_name, out_name):
  """Substraction and squared L2 of signals from source node u and target node v.

  choose the signals on source node, substract signals from target nodes. Applys
  squared L2-norm afterwards.

  Args:
    u_name: attribute name on source node
    v_name: attribute name on target node
    out_name: where to save the results

  Returns:
    exec_op: operation to be executed is message-passing
  """

  def exec_op(edges):
    """Function warpper.

    creates a function to be applied later during message passing.

    Args:
      edges: container of information on the edges

    Returns:
      results: updated dictionary with the results
    """
    feat_dst = edges.src[u_name]
    feat_src = edges.dst[v_name]
    axes = range(len(feat_dst.shape))[1:]
    sum_over_dim = tuple(axes)
    diff = (feat_dst - feat_src)
    sqr_values = torch.pow(diff, 2)
    l2_norm = sqr_values.sum(dim=sum_over_dim)
    results = {out_name: l2_norm}
    return results

  return exec_op


def reduce_sum(nodes):
  """Compute the sum over incoming signals.

  Look at the incomining signals in the mailbox of nodes, and compute the sum
  over all incoming signals over existin g edges.

  Args:
    nodes: container of nodes information.

  Returns:
    results: dictionary with results
  """
  msg = nodes.mailbox['grad_score']
  supp_size = nodes.data['supp_size']
  msg_sum = torch.sum(msg, dim=1)
  supp_size = supp_size.view(msg_sum.shape)
  v_hat = msg_sum / supp_size

  results = {'v_hat': v_hat}
  return results


def argmax(from_edge, save_at_node, max_scores_name):
  """Compute the argmax and max .

  Take the max and argmax over all incoming signals over the edges.

  Args:
    from_edge: attribute name on source node
    save_at_node: where to save the max score
    max_scores_name: where to save the argmax

  Returns:
    exec_op: operation to be executed is message-passing
  """

  def arg_max_op(nodes):
    """Function warpper.

    creates a function to be applied later during message passing.

    Args:
      nodes: container of information on the nodes

    Returns:
      results: updated dictionary with the results
    """
    attn = nodes.mailbox[from_edge]
    max_scores, arg_max_score = attn.max(dim=1)

    max_scores = max_scores.unsqueeze(dim=1)
    arg_max_score = arg_max_score.unsqueeze(dim=1)
    results = {save_at_node: arg_max_score, max_scores_name: max_scores}
    return results

  return arg_max_op


def choose_based_on(from_edge, max_indices_name, save_at_node):
  """Choose values based the predefined keys.

  choose values based on keys. It can e.g. be combined with argmax to choose
  the maximum values

  Args:
    from_edge: attribute name on source node
    max_indices_name: indices of the max values
    save_at_node: where to save the results

  Returns:
    exec_op: operation to be executed is message-passing
  """

  def choose_op(nodes):
    """Function warpper.

    creates a function to be applied later during message passing.

    Args:
      nodes: container of information on the nodes

    Returns:
      results: updated dictionary with the results
    """
    scores = nodes.mailbox[from_edge]
    max_indices = nodes.data[max_indices_name]
    assert scores.shape[0] == max_indices.shape[
        0], 'how to take only scores for corresponding edges again?'
    assert len(scores.shape) == 4
    max_elements = torch.gather(scores, 1, max_indices)
    results = {save_at_node: max_elements}
    return results

  return choose_op
