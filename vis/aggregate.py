"""Reads results from automl logs."""

import hpbandster.core.result as hpres
import sys
import os
import numpy as np
from data_reader.data_utils import append_pickle


def flatten(perf_dict):
  """Flattens lists of results.

  Args:
      perf_dict (dict): Performance of all runs.

  Returns:
      results: array of all results
  """
  results_list = []
  for k, runs in perf_dict.items():
    c_results = [r[1] for r in runs[0]]
    results_list.extend(c_results)
  results = np.array(results_list)
  return results


def merge_results(all_runs, id2conf):
  """Merges results into a dict.

  Args:
      all_runs : results of all runs.
      id2conf : mapping from run_id to configuration.

  Returns:
      TYPE: Description
  """
  merged_results = {}
  for config in all_runs:
    config_id = config['config_id']
    if config['info'] is not None:
      merged_results[config_id] = {}
      merged_results[config_id]['info'] = config['info']
      merged_results[config_id]['info']['budget'] = config['budget']
      merged_results[config_id]['config'] = id2conf[config_id]['config']

  assert len(merged_results) > 0, 'insertions not successful'
  return merged_results


def create_key(config_value, separate_over):
  """Create keys based on configuration.

  Args:
      config_value: value of the configurations.
      separate_over: Keys to separate over.

  Returns:
      Key: New key.
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
  return key


def separate_results(merged_results, separate_over, max_over):
  """Separates results based on the keys.

  Args:
      merged_results : merged_results3
      separate_over : key to separate over.
      max_over : key tot perform the max-operation over.

  Returns:
      dict: organized into separated keys and values.
  """
  separating_dicts = {}
  for config_id, config_value in merged_results.items():
    key = create_key(config_value, separate_over)
    max_over_key = create_key(config_value, max_over)
    c_dict_sep_over = separating_dicts.get(key, {})

    # one more level for max_over:
    config_results_dict = c_dict_sep_over.get(max_over_key, {})
    config_results_dict[config_id] = config_value
    c_dict_sep_over[max_over_key] = config_results_dict
    separating_dicts[key] = c_dict_sep_over
  return separating_dicts


def get_all_values(keyword, max_over_value):
  """Retrieves all values based on keyword.

  Args:
      keyword (str): keyword.
      max_over_value (str): value to perform max-over.

  Returns:
      values: found values.
  """
  values = [
      run_value['info'][keyword]
      for run_id, run_value in max_over_value.items()
  ]
  return values


def aggreate(separated_results, file_name=None):
  """Aggreate results into a dictionary.

  Args:
      separated_results: separated results.
      file_name (None, optional): Description.

  Returns:
      results: all results.
  """
  aggreated_results = {}
  for chunk_id, chunk_value in separated_results.items():
    for max_over_id, max_over_value in chunk_value.items():
      test_values = get_all_values('best_test_perf', max_over_value)
      test_values_std = get_all_values('best_test_perf_std', max_over_value)

      valid_values = get_all_values('best_valid_perf', max_over_value)

      if len(test_values) == 0:
        continue

      aggreated_results[chunk_id] = aggreated_results.get(chunk_id, {})
      aggreated_results[chunk_id][max_over_id] = aggreated_results[
          chunk_id].get(max_over_id, {})
      current_dict = aggreated_results[chunk_id][max_over_id]

      # test values agg
      test_values = np.array(test_values)
      valid_values = np.array(valid_values)
      test_values_std = np.array(test_values_std)

      best_entry = valid_values.argmax()

      current_dict['test_value'] = test_values
      current_dict['test_max'] = test_values[best_entry]
      current_dict['len'] = len(test_values)
      current_dict['test_std'] = test_values_std[best_entry]
      # valid values agg
      current_dict['valid_value'] = valid_values
      current_dict['valid_max'] = valid_values[best_entry]
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
  """Search for max values.

  Args:
      aggreated_results (dict): collected results.

  Returns:
      resutls: found results.
  """
  results = {}
  for chunk_id, chunk_value in aggreated_results.items():
    reduced_dict = {k: v['valid_max'] for k, v in chunk_value.items()}
    max_key = max(reduced_dict, key=reduced_dict.get)
    value = chunk_value[max_key]

    results[chunk_id] = {
        'n_runs': value['len'],
        'test_max': value['test_max'],
        'test_std': value['test_std'],
        'max_key': max_key
    }

  assert len(results) > 0
  return results


def get_reduced_key(chunk_id, last_sep_over):
  """Constructs a reduced key.

  Args:
      chunk_id (str): keyword of chunk.
      last_sep_over (str): separate at the end over these keys.

  Returns:
      key: reduced_key.
  """
  key_value_pairs = chunk_id.split('-')
  kv_mappings = {}
  for kv in key_value_pairs:
    kv_list = kv.split(':')
    if len(kv_list) > 1:
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
  return key


def find_all_matching_keys(all_results, chunk_id):
  """Pattern-match keys.

  Args:
      all_results (dict): all collected results.
      chunk_id (str): current chunk of interest.

  Returns:
      matching_keys: all keys which match chunk_id.
      cleaned_key: cleaned key.
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

  assert len(matching_keys) > 0
  assert cleaned_key != ''
  assert len(cleaned_key) > 0
  return matching_keys, cleaned_key


def output_overall(aggreated_results, last_sep_over):
  """Sorts the outputs and prints them into std.

  Args:
      aggreated_results (dict): collected results.
      last_sep_over (dict): separation over keys.
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

  for out in outputs:
    print(out)


def analyze(experiment_name, separate_over, max_over):
  """Anayze the run experiments.

  Args:
      experiment_name (str): Name and location of the experiments
      separate_over (dict): keys to perform advanced analyses.
      max_over (dict): keys to perform advanced analyses.
  """
  base_path = '/mnt/datasets/'
  path = os.path.join(base_path, experiment_name)
  result = hpres.logged_results_to_HBS_result(path)
  file_name = 'runs_results.json'
  if os.path.exists(file_name):
    os.remove(file_name)

  all_runs = result.get_all_runs()
  id2conf = result.get_id2config_mapping()

  merged_results = merge_results(all_runs, id2conf)
  separated_results = separate_results(merged_results, separate_over, max_over)
  aggreated_results = aggreate(separated_results, file_name)

  max_perf_in_each_chunk = get_max_values_for_each_run(aggreated_results)
  output_overall(max_perf_in_each_chunk, separate_over)


if __name__ == '__main__':
  """Runs the script using these options."""

  experiment_name = sys.argv[1]

  separate_over = [
      'model',
      'ignore_last_att_weights',
      'pooling_residual',
      'transform',
      'train_using_pseudo_labels',
      'data',
      'samples_per_class',
      'repeated_runs',
  ]
  max_over = [
      'num_layers', 'attn_drop', 'in_drop', 'dropout', 'learning_rate',
      'attention_weight', 'lambda_sparsemax', 'alpha_sparsemax'
      'att_last_layer', 'num_hidden'
  ]

  analyze(experiment_name, separate_over, max_over)
