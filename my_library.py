def test_load():
  return 'loaded'
  
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]
  
def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01 
  
def cond_probs_product(table, evidence_row, target, target_value):
  #your function body below
  table_columns = up_list_column_names(table)
  evidence_columns =  table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)
  cond_prob_list = [cond_prob (table, i, j, target, target_value)for i, j in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator
  
def prior_prob (table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a
  
def naive_bayes(table, evidence_row, target):
  target_value = 0
  num_0 = cond_probs_product(table,evidence_row,target,target_value) * prior_prob(table, target, target_value)
  target_value = 1
  num_1 = cond_probs_product(table,evidence_row,target,target_value) * prior_prob(table, target, target_value)
  neg, pos = compute_probs(num_0,num_1)
  return[neg, pos]

def metrics(zipped_list):
  predictions, labels = zip(*zipped_list)
  all_cases = len(predictions)
  tp = sum(p == 1 and l == 1 for p, l in zipped_list)
  fn = sum(p == 0 and l == 1 for p, l in zipped_list)
  fp = sum(p == 1 and l == 0 for p, l in zipped_list)
  tn = sum(p == 0 and l == 0 for p, l in zipped_list)
  precision = tp / (tp + fp) if tp + fp != 0 else 0
  recall = tp / (tp + fn) if tp + fn != 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
  accuracy = (tp + tn) / all_cases
  metrics_dict = {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}
  return metrics_dict

def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)
  for architecture in architectures:
    all_results = up_neural_net(train_table, test_table, architecture, target)
    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]
    print(f'Architecture: {architecture}')
    print(up_metrics_table(all_mets))
  return None
