"""Example 6 -  Analysis of a Run ==============================

This example takes a run from example 5 and performs some analysis of it.
It shows how to get the best performing configuration, and its attributes.
More advanced analysis plots provide some insights into a run and the problem.

"""

import hpbandster.core.result as hpres
import sys
import os
import numpy as np
from data_reader.data_utils import append_pickle
"""Protein:

    - Mean over:
        - normal_class
    - Separated:
        - model
        - data
        - Learning_rate
        - num_layers
        - train_dev_proportion
        - anomaly proportion.
citation_graphs:
    - exact the same thing
Implmentation:
    - create new dict wtih proper keys for Separation (chunks)
    - then in each chunk, aggreate over all similar models
    - convert their performance to numy array & output means.
    - sort outputs and print to some file (or console)
Additionally
    - output how many runs we have got
    - std

"""


def compute_corrcoef(a1, a2):
  """Summary

  Args:
      a1 (TYPE): Description
      a2 (TYPE): Description

  Returns:
      TYPE: Description
  """
  cov = np.corrcoef(a1, a2)[0][1]
  return cov


def flatten(perf_dict):
  """Summary

  Args:
      perf_dict (TYPE): Description

  Returns:
      TYPE: Description
  """
  #lcs[(0, 0, 1)][0][..][1]
  results_list = []
  for k, runs in perf_dict.items():
    c_results = [r[1] for r in runs[0]]
    results_list.extend(c_results)
  results = np.array(results_list)
  return results


def merge_results(all_runs, id2conf):
  """merge and filter results @params: id2conf: dict.

  {id:{'config':{}}}
          @params: all_runs: dict. {config_id: (0, 0, 0)   budget: 400.000
          loss: 0.994
          time_stamps: 0.0 (), 0.000 (started), 173.571 (finished)
          info: {'best_epoch': 0, 'best_test_perf': 0.442,
          'best_test_perf_f1': 0.761, 'best_valid_perf': 0.547,
          'best_valid_perf_f1': 0.005}
          }

  Args:
      all_runs (TYPE): Description
      id2conf (TYPE): Description

  Returns:
      TYPE: Description
  """
  merged_results = {}
  for config in all_runs:
    config_id = config['config_id']
    # config = run_value['config']
    if config['info'] is not None:  #and config['loss'] is not 1:
      # add to the new dict
      merged_results[config_id] = {}
      merged_results[config_id]['info'] = config['info']
      merged_results[config_id]['info']['budget'] = config['budget']
      merged_results[config_id]['config'] = id2conf[config_id]['config']

  assert len(merged_results) > 0, 'insertions not successful'
  return merged_results


def create_key(config_value, separate_over):
  """Summary

  Args:
      config_value (TYPE): Description
      separate_over (TYPE): Description

  Returns:
      TYPE: Description
  """
  key = ''
  for keyword in separate_over:
    if keyword in config_value['config']:
      value = config_value['config'][keyword]
      if type(value) == float:
        key += '{}:{:.2}-'.format(keyword, value)
      if type(value) == int:
        key += '{}:{:02d}-'.format(keyword, value)
      else:
        key += '{}:{}-'.format(keyword, value)
    #else:
    #print('keyword missing {}'.format(keyword))
    #pdb.set_trace()
  return key


def separate_results(merged_results, separate_over, max_over):
  """Summary

  Args:
      merged_results (TYPE): Description
      separate_over (TYPE): Description
      max_over (TYPE): Description

  Returns:
      TYPE: Description
  """
  separating_dicts = {}
  for config_id, config_value in merged_results.items():
    key = create_key(config_value, separate_over)
    #print(key)
    #pdb.set_trace()
    max_over_key = create_key(config_value, max_over)
    c_dict_sep_over = separating_dicts.get(key, {})

    # one more level for max_over:
    config_results_dict = c_dict_sep_over.get(max_over_key, {})
    config_results_dict[config_id] = config_value
    c_dict_sep_over[max_over_key] = config_results_dict
    ###########################
    separating_dicts[key] = c_dict_sep_over
  return separating_dicts


def get_all_values(keyword, max_over_value, threshold):
  """Summary

  Args:
      keyword (TYPE): Description
      max_over_value (TYPE): Description
      threshold (TYPE): Description

  Returns:
      TYPE: Description
  """
  values = [
      run_value['info'][keyword]
      for run_id, run_value in max_over_value.items()
      if run_value['info'][keyword] > threshold
  ]
  return values


def aggreate(separated_results, threshold, file_name=None):
  """Summary

  Args:
      separated_results (TYPE): Description
      threshold (TYPE): Description
      file_name (None, optional): Description

  Returns:
      TYPE: Description
  """
  aggreated_results = {}
  for chunk_id, chunk_value in separated_results.items():
    # for keyword in mean_over:
    #all_key_word_values = {run_value['config'][keyword] for _,run_value in chunk_value.items()}
    #for keyword_value in all_key_word_values:
    for max_over_id, max_over_value in chunk_value.items():
      try:
        # Elliptic dataset has this final_test_perf
        test_values = get_all_values('final_test_perf', max_over_value,
                                     threshold)
      except Exception as e:
        test_values = get_all_values('best_test_perf', max_over_value,
                                     threshold)
        test_values_std = get_all_values('best_test_perf_std', max_over_value,
                                         threshold)

        valid_values = get_all_values('best_valid_perf', max_over_value,
                                      threshold)

      if len(test_values) == 0:
        continue

      aggreated_results[chunk_id] = aggreated_results.get(chunk_id, {})
      aggreated_results[chunk_id][max_over_id] = aggreated_results[
          chunk_id].get(max_over_id, {})
      current_dict = aggreated_results[chunk_id][max_over_id]

      # if len(test_values) > 1:
      #     print('only take the max valuess.')
      #     pdb.set_trace()
      # test values agg
      test_values = np.array(test_values)
      valid_values = np.array(valid_values)
      test_values_std = np.array(test_values_std)

      best_entry = valid_values.argmax()

      current_dict['test_value'] = test_values
      current_dict['test_max'] = test_values[best_entry]
      # current_dict['test_max'] = test_values.min()
      current_dict['len'] = len(test_values)
      current_dict['test_std'] = test_values_std[best_entry]
      # valid values agg
      current_dict['valid_value'] = valid_values
      current_dict['valid_max'] = valid_values[best_entry]
      # current_dict['valid_max'] = valid_values.min()
      current_dict['valid_len'] = len(valid_values)
      current_dict['valid_std'] = valid_values.std()
      assert current_dict['valid_len'] == current_dict['len']

      dict_for_printing = {
          'config_name': chunk_id,
          'test_max': current_dict['test_max'],
          'test_std': current_dict['test_std']
      }
      append_pickle(file_name, dict_for_printing)

  return aggreated_results


def get_max_values_for_each_run(aggreated_results):
  """Summary

  Args:
      aggreated_results (TYPE): Description

  Returns:
      TYPE: Description
  """
  results = {}
  for chunk_id, chunk_value in aggreated_results.items():
    # for keyword, value in chunk_value.items():
    reduced_dict = {k: v['valid_max'] for k, v in chunk_value.items()}

    #print('{}: max over {} runs'.format(chunk_id, len(reduced_dict)))
    max_key = max(reduced_dict, key=reduced_dict.get)
    value = chunk_value[max_key]

    results[chunk_id] = {
        'n_runs': value['len'],
        'test_max': value['test_max'],
        #'test_mean' : value['test_mean'],
        'test_std': value['test_std'],
        'max_key': max_key
    }

  assert len(results) > 0
  return results


def get_reduced_key(chunk_id, last_sep_over):
  """Summary

  Args:
      chunk_id (TYPE): Description
      last_sep_over (TYPE): Description

  Returns:
      TYPE: Description
  """
  key_value_pairs = chunk_id.split('-')
  kv_mappings = {}
  for kv in key_value_pairs:
    kv_list = kv.split(':')
    # print(kv)
    if len(kv_list) > 1:
      # ignore the non-parseable, list entry
      k = kv_list[0]
      v = kv_list[1]
      kv_mappings[k] = v
  assert len(kv_mappings) > 0
  key = ''
  for keyword in last_sep_over:
    if keyword in kv_mappings:
      new_key = '{}:{}-'.format(keyword, kv_mappings[keyword])
      key += new_key
    else:
      assert keyword not in chunk_id, 'Parsing might be wrong {}'.format(
          keyword)
  assert key != ''
  #print(key)
  return key


def find_all_matching_keys(all_results, chunk_id):
  """Summary

  Args:
      all_results (TYPE): Description
      chunk_id (TYPE): Description

  Returns:
      TYPE: Description
  """
  pattern = 'repeated_runs:0'
  start_pos = chunk_id.index(pattern)
  end_pos = start_pos + len(pattern) + 1
  contained_pattern = chunk_id[start_pos:end_pos]
  matching_keys = []
  for i in range(10):
    new_pattern = pattern + str(i)
    new_string = chunk_id.replace(contained_pattern, new_pattern)
    matching_keys.append(new_string)

  cleaned_key = chunk_id[:start_pos] + chunk_id[end_pos:]

  # print(matching_keys)
  # print(cleaned_key)
  assert len(matching_keys) > 0
  assert cleaned_key != ''
  assert len(cleaned_key) > 0
  return matching_keys, cleaned_key


def output_overall(aggreated_results, last_sep_over):
  """Summary

  Args:
      aggreated_results (TYPE): Description
      last_sep_over (TYPE): Description
  """
  # gather the data
  all_data = {}
  for chunk_id, chunk_value in aggreated_results.items():
    key = get_reduced_key(chunk_id, last_sep_over)
    c_results = all_data.get(key, [])
    c_results.append(chunk_value['test_max'])
    all_data[key] = c_results

  # gather perf strings
  all_results = {}
  for chunk_id, chunk_value in all_data.items():
    test_values_array = np.array(chunk_value)
    # if len(test_values_array) > 1:
    # pdb.set_trace()
    assert len(test_values_array) == 1
    test_perf = test_values_array[0]
    max_key = aggreated_results[chunk_id]['max_key']
    all_results[chunk_id] = test_perf

  final_results = {}
  outputs = []
  for chunk_id in all_results:

    matching_keys, cleaned_key = find_all_matching_keys(all_results, chunk_id)
    if cleaned_key not in final_results:

      c_final_results_ = [
          all_results[key] for key in matching_keys if key in all_results
      ]
      # pdb.set_trace()
      c_final_results_ = np.array(c_final_results_)
      final_results[cleaned_key] = c_final_results_.mean()
      results_str = '{0}__bestConfig_{3} mean {1:.2f} std {2:.4f}'.format(
          cleaned_key, final_results[cleaned_key] * 100,
          aggreated_results[chunk_id]['test_std'], max_key)
      outputs.append((results_str))

  outputs = sorted(outputs)
  print('################################################')
  print('ALL Result in average')
  print('################################################')

  for o in outputs:
    print(o)


def filter(all_runs, num_layers):
  """Summary

  Args:
      all_runs (TYPE): Description
      num_layers (TYPE): Description

  Returns:
      TYPE: Description
  """
  if num_layers != -1:
    constraints = {
        'num_layers': num_layers,
        # 'model': 'gat',
    }
  else:
    constraints = {}
  filtered_runs = {}
  for run_id, run_value in all_runs.items():
    if run_value['info']:
      all_satified = True
      for c_name, c_value in constraints.items():
        if c_name in run_value['config']:
          satified = run_value['config'][c_name] == c_value
          all_satified = all_satified and satified
      if all_satified:
        c_id = run_id
        filtered_runs[c_id] = run_value
      else:
        continue
    else:
      continue

  return filtered_runs


def analyze(experiment_name, mean_over, separate_over, threshold, max_over,
            last_sep_over, num_layers):
  """Summary

  Args:
      experiment_name (TYPE): Description
      mean_over (TYPE): Description
      separate_over (TYPE): Description
      threshold (TYPE): Description
      max_over (TYPE): Description
      last_sep_over (TYPE): Description
      num_layers (TYPE): Description
  """
  base_path = '/mnt/datasets/'
  path = os.path.join(base_path, experiment_name)
  result = hpres.logged_results_to_HBS_result(path)
  file_name = 'runs_results.json'
  if os.path.exists(file_name):
    os.remove(file_name)

  # get all executed runs
  all_runs = result.get_all_runs()
  # get the 'dict' that translates config ids to the actual configurations
  id2conf = result.get_id2config_mapping()

  merged_results = merge_results(all_runs, id2conf)
  merged_results = filter(merged_results, num_layers)
  separated_results = separate_results(merged_results, separate_over, max_over)
  aggreated_results = aggreate(separated_results, threshold, file_name)

  max_perf_in_each_chunk = get_max_values_for_each_run(aggreated_results)
  output_overall(max_perf_in_each_chunk, last_sep_over)

  # output_detailed(max_perf_in_each_chunk)


if __name__ == '__main__':
  if len(sys.argv) == 4:
    experiment_name = sys.argv[1]
    threshold = float(sys.argv[2])
    num_layers = float(sys.argv[3])
  else:
    assert False, 'please provide arguments'
    experiment_name = 'citation_graphs/eval_yelp/'
    threshold = 0
    num_layers = -1

  mean_over = [
      'normal_class',  # 0..3
      'run_multiple_times'  # 0..3
  ]
  separate_over = [
      'model',
      'ignore_last_att_weights',
      'pooling_residual',
      'transform',
      'pseudo_label_mode',
      'pseudo_labels_loss',
      'pseudo_labels_eval',
      'pseudo_labels_learning_detach',
      'prop_at_training',
      'prop_at_inference',
      'train_using_pseudo_labels',
      'data',
      # 'attention_loss',
      # 'use_provided_data_split',
      # 'train_dev_proportion',
      # 'anomaly_proportion',
      # 'num_heads',
      # 'jknet_aggregator_type',
      # 'layer_aggregation',
      # 'graphsage_aggregator_type',
      # 'main_loss',
      'samples_per_class',
      'repeated_runs',
  ]
  max_over = [
      'num_layers', 'attn_drop', 'in_drop', 'dropout', 'learning_rate',
      'attention_weight', 'lambda_sparsemax', 'alpha_sparsemax'
      'att_last_layer', 'num_hidden'
  ]

  last_sep_over = separate_over
  analyze(experiment_name, mean_over, separate_over, threshold, max_over,
          last_sep_over, num_layers)
