import re
import pandas as pd
import numpy as np

class EvaluationVehicleDataset:
    """
    Evaluation class for OSCAR on DTC sequences
    """
    def __init__(self, ep_def_url: str, label_mapping: dict, tokenizer, 
                verbose: bool=False, consider_prevalence: bool= False, consider_not_present: bool=True):
        self.label_mapping = label_mapping
        ep_def = pd.read_parquet(ep_def_url)[['dtc_logic_string', 'error_pattern_name']]
        self._ep_def = ep_def[ep_def['error_pattern_name'].isin(list(label_mapping.keys()))]
        self.tokenizer = tokenizer
        self.consider_prevalence = consider_prevalence 
        self.consider_not_present = consider_not_present
        print(f"Considering prevalence strength to be discovered {consider_prevalence}, Considering detecting no present causes in input {consider_not_present}")
        if verbose:
            print(f"label mapping len {len(label_mapping)}, len in ep def {len(ep_def_)}")
        self.verbose = verbose
        # Metric per label with its samples 
        self.metrics = {}
        self.fn_et_ci = {}
        self.total_samples = 0
        self.total_labels = 0
        self.total_not_present_in_input_ids = 0

    def extract_causes_from_logic(self, logic):
        """
        Extract each elements of the Markov Boundary.
        
        It needs to be updated with the anonymized version.
        """
        # Define the separators you want to split on
        # For example, we want to split on commas, spaces, exclamation marks, hyphens, semicolons, and colons
        separators = r"[&|]+"  # Regular expression pattern for multiple separators

        # Split the string using re.split()
        result = re.split(separators, logic)

        # Remove empty strings from the result (if any)
        result = [s.replace(' ', '').replace('(', '').replace(')', '') for s in result if s]
        if not self.consider_prevalence:
            if self.verbose:
                print("We will not consider prevalence, thus filtered list")
            result = [s for s in result if not s.startswith('!')]
        if self.verbose:
            print("Resulted extracted dtc logic arr: ", result)
        return result

    def get_present_cause_from_inputs_ids(self, dtc_logic, input_ids):
        """
        Extract the causes present in the input_ids
        
        It needs to be updated with the anonymized dataset version.
        """
        actual_cause_present_in_input_ids = []
        for dtc_cause in input_ids.split(' '):
            for splited_dtc in dtc_logic:
                if splited_dtc.upper() in dtc_cause.upper():
                    actual_cause_present_in_input_ids.append(dtc_cause)

        return actual_cause_present_in_input_ids

    def calculate_precision_recall_f1(self, inferred_causes, true_causes):
        """
        Calculate precision, recall, and F1-score for an inferred Markov boundary.

        :param inferred_causes: List of inferred causal tokens.
        :param true_causes: List of actual causal tokens.
        :return: Dictionary with precision, recall, F1-score, TP, FP, FN.
        """
        inferred_set, true_set = set(inferred_causes), set(true_causes)
        true_positives = len(inferred_set & true_set)  # Intersection
        false_positives = len(inferred_set - true_set)  # Inferred but incorrect
        false_negatives = len(true_set - inferred_set)  # Missed true causes

        precision = true_positives / len(inferred_set) if inferred_set else 0.0
        recall = true_positives / len(true_set) if true_set else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        
        return {
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
        }

    def weighted_average(self):
        """
        Label contributes proportionally to its count.
        """
        total_precision = 0
        total_recall = 0
        total_f1_score = 0
        total_h_score = 0
        num_samples = self.total_samples - self.total_not_present_in_input_ids  # Use actual sample count

        for label, values in self.metrics.items():
            weight = values['count'] / num_samples  # Weight based on sample proportion
            avg_precision_for_label = values['precision']/values['count']
            avg_recall_for_label = values['recall']/values['count']
            avg_f1_for_label = values['f1_score']/values['count']
            
            total_precision += avg_precision_for_label*weight
            total_recall += avg_recall_for_label * weight
            total_f1_score += avg_f1_for_label*weight

        return {
            'precision': round(total_precision*100, 3),
            'recall': round(total_recall*100, 3),
            'f1_score': round(total_f1_score*100, 3),
            '%_not_present': round((self.total_not_present_in_input_ids / self.total_samples) * 100, 3)
        }

    def micro_average(self):
        """
        Micro-average: Computes metrics globally by summing up TP, FP, FN.
        """
        total_tp = self.fn_et_ci['g']['tp']
        total_fp = self.fn_et_ci['g']['fp']
        total_fn = self.fn_et_ci['g']['fn']

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'precision': round(precision * 100, 3),
            'recall': round(recall * 100, 3),
            'f1_score': round(f1_score * 100, 3),
            '%_not_present': round((self.total_not_present_in_input_ids / self.total_samples) * 100, 3)
        }

    def macro_average(self):
        """
        Each label contribute equally
        """
        total_precision = 0
        total_recall = 0
        total_f1_score = 0
        num_labels = len(self.metrics)
        print("num labels", num_labels)

        for values in self.metrics.values():
            total_precision += values['precision']/values['count']
            total_recall += values['recall']/values['count']
            total_f1_score += values['f1_score']/values['count']

        return {
            'precision': round(total_precision / num_labels *100, 3),
            'recall': round(total_recall / num_labels*100, 3),
            'f1_score': round(total_f1_score / num_labels*100, 3),
            'num_labels': num_labels,
            '%_not_present': round((self.total_not_present_in_input_ids/self.total_samples) * 100, 3)
        }

    def is_input_ids_present_on_rule_for_label(self, input_ids: str, label_name: dict):
        try:
            dtc_logic_for_y_c = self._ep_def[self._ep_def['error_pattern_name'] == label_name]['dtc_logic_string'].iloc[0]
            if self.verbose:
                print(f"DTC Logic for label {label_name} is {dtc_logic_for_y_c}")
        except:
            if self.verbose:
                print(f"tested Label name {label_name} not present in ep def")
            return False
        
        splited_dtc_logic_for_y_c = self.extract_causes_from_logic(dtc_logic_for_y_c)
        present_in_input_ids = self.get_present_cause_from_inputs_ids(splited_dtc_logic_for_y_c, input_ids)
  
        if (len(present_in_input_ids) == 0) and (self.consider_not_present == False):
            if self.verbose:
                print(f"Nothing is present in the input_ids {present_in_input_ids}, passing ..")
            self.total_not_present_in_input_ids +=1
            self.total_samples+=1
            return False
        
    def eval_on_sample(self, input_ids: str, y_c: int, label_name: str, cause_label: dict):
        """
        For one sample, perform precision, recall, f1 score evaluation
        causes: dict is a nested dict
        """
        try:
            dtc_logic_for_y_c = self._ep_def[self._ep_def['error_pattern_name'] == label_name]['dtc_logic_string'].iloc[0]
            if self.verbose:
                print(f"DTC Logic for label {label_name} is {dtc_logic_for_y_c}")
        except:
            if self.verbose:
                print(f"tested Label name {label_name} not present in ep def")
            return False

        splited_dtc_logic_for_y_c = self.extract_causes_from_logic(dtc_logic_for_y_c)
        present_in_input_ids = self.get_present_cause_from_inputs_ids(splited_dtc_logic_for_y_c, input_ids)

        dtc_mb_extracted_with_type = [self.tokenizer.decode(mb_i) for mb_i in cause_label]
        if self.verbose:
            print(" present in inputs ids:", present_in_input_ids, " infered", dtc_mb_extracted_with_type)

        if (len(present_in_input_ids) == 0) and (self.consider_not_present == False):
            if self.verbose:
                print(f"Nothing is present in the input_ids {present_in_input_ids}, passing ..")
            self.total_not_present_in_input_ids +=1
            self.total_samples+=1
            return False

        o = self.calculate_precision_recall_f1(dtc_mb_extracted_with_type, present_in_input_ids)
        
        if self.verbose:
            print("Metrics: ", o)

        if y_c not in self.metrics:
            self.fn_et_ci['g'] = {'tp': o['tp'], 'fp': o['fp'], 'fn': o['fn'], 'count': 1}
            self.metrics[y_c] = {'precision': o['precision'], 'recall': o['recall'], 'f1_score': o['f1_score'], 'count': 1}
        else:
            self.fn_et_ci['g'] = {'tp': self.fn_et_ci['g']['tp']+o['tp'], 'fn': self.fn_et_ci['g']['fn']+o['fn'], 'fp': self.fn_et_ci['g']['fp']+o['fp'], 'count': self.fn_et_ci['g']['count']+1}
            self.metrics[y_c] = {'precision': self.metrics[y_c]['precision']+o['precision'], 
                                        'recall': self.metrics[y_c]['recall']+o['recall'], 
                                        'f1_score': self.metrics[y_c]['f1_score']+o['f1_score'],
                                        'count': self.metrics[y_c]['count']+1,
                                   
        self.total_samples+=1