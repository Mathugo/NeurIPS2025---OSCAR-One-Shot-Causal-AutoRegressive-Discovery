import sys, torch, json, random, os
from datetime import datetime
from eval_oscar import EvaluationVehicleDataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from torch.cuda.amp import autocast  # Mixed precision
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import numpy as np
from itertools import combinations
from accelerate import Accelerator

""" CHANGE THIS TO LOAD YOUR PRETRAINED TFx TFy 
To run OSCAR you need two pretrained Tfx, Tfy (see the paper). Optionnaly a tokenizer to get the decoded node name.
"""
sys.path.append('../../src/')
sys.path.append('..')
print("Please input your pretrained model path directly in the script to run OSCAR")
from ..tfx.pretraining import CarFormerForPretraining
from ..tfy.model import EPCausalPredictorRotCrossLlamaAttention
tfx = CarFormerForPretraining.from_pretrained('.')
tfy = EPCausalPredictorRotCrossLlamaAttention.from_pretrained('.')
# if you have a tokenizer to get the decoded token id
tokenizer = AutoTokenizer.from_pretrained('.')
from core.epredictor.metrics import next_ccm_time_custom_collate as fn_collate

# Load the event sequence dataset
s_test = load_dataset("parquet", data_files=f"data/ds_test.parquet")['train']

# print
VERBOSE = False
PRINT = False

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

def topk_p_sampling(z, prob_x, c: int, n: int = 128, p: float = 0.9, k: int = 10,
                       cls_token_id: int = 1, temp: float = None):
    # Step 1: restrict to vocab subset
    input_ = prob_x[:, :c]

    # Step 2: Temperature scaling
    if temp is not None:
        logits = torch.log(input_ + 1e-8)
        input_ = torch.softmax(logits / temp, dim=-1)

    # Step 3: Top-k filter
    topk_values, topk_indices = torch.topk(input_, k=k, dim=-1)

    # Step 4: Top-p filter over top-k values
    sorted_probs, sorted_idx = torch.sort(topk_values, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens beyond cumulative probability p
    mask = cum_probs > p
    
    # Ensure at least one token is kept
    mask[..., 0] = 0

    # Mask and normalize
    filtered_probs = sorted_probs.masked_fill(mask, 0.0)
    filtered_probs += 1e-8  # for numerical stability
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

    # Unscramble to match original top-k indices
    # Need to reorder the sorted indices back to original top-k
    reorder_idx = torch.argsort(sorted_idx, dim=-1)
    filtered_probs = torch.gather(filtered_probs, -1, reorder_idx)

    # Step 5: Sampling
    batched_probs = filtered_probs.unsqueeze(1).repeat(1, n, 1, 1)        # (bs, n, seq_len, k)
    batched_indices = topk_indices.unsqueeze(1).repeat(1, n, 1, 1)        # (bs, n, seq_len, k)

    sampled_idx = torch.multinomial(batched_probs.view(-1, k), 1)         # (bs*n*seq_len, 1)
    sampled_idx = sampled_idx.view(-1, n, c).unsqueeze(-1)

    sampled_tokens = torch.gather(batched_indices, -1, sampled_idx).squeeze(-1)
    sampled_tokens[..., 0] = cls_token_id

    # Reconstruct full sequence
    z_expanded = z.unsqueeze(1).repeat(1, n, 1)[..., c:]
    return torch.cat((sampled_tokens, z_expanded), dim=-1)

def OSCAR_parallel(tfx, tfy, params, ds_test, accelerator, min_context: int=20):
    """
    Perform One-shot multi label causal discovery on a data loader composed of batches of sequential data
    """
    print("Loading test dataset ..")
    start_time = datetime.now()
    
    if params.get('multiple_gpus', False):
        print("Using sampler and multiple GPUS")
        sampler = DistributedSampler(ds_test, seed=42, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True)
    else:
        sampler = None
    dl_test = DataLoader(
    ds_test,
    batch_size=params['BS'],
    sampler=sampler,
    collate_fn=fn_collate
    )
    # Move models to the right device
    tfy, dl_test = accelerator.prepare(tfy, dl_test)
    # force the cast to fp32
    tfx = tfx.to(accelerator.device)
    
    if not params['consider_prevalence']:
        print("[!] Prevalence strength (cs < 0) will not be considered")

    eps = params.get('eps', 1e-8)
    print("Eps for numerical stability: ", eps)
    c = min_context
    total_step = 0
    thresholds = {}
    if params['sampling'].get('type',False):
        print(f"Sampling enabled with type {params['sampling'].get('type', None)}")
        print(f"Real BS with Sampling is {params['BS']*params['sampling'].get('value', 64)}")
    
    if accelerator.is_main_process:
        dl_test = tqdm(dl_test, desc="Inference")
    else:
        dl_test = dl_test
    eval_o = params['eval_o']
    labels_ = {}
    with torch.no_grad():
        # Perform batch enumeration and loop through the DataLoader
        for batch_index, batch in enumerate(dl_test):
            gathered_items = accelerator.gather_for_metrics(batch)

            o_b = tfx(attention_mask=batch['attention_mask'], input_ids=batch['input_ids'], time=batch['time'], 
                      mileage=batch['mileage'])
            hidden_states = o_b['prediction_logits']
            prob_x = torch.nn.functional.softmax(hidden_states, dim=-1)

            # We can compute the expected value 
            num_samples = params['sampling'].get('value', 64)
            expanded_attention_mask = batch['attention_mask'].unsqueeze(1).repeat(1, num_samples, 1)
            expanded_time = batch['time'].unsqueeze(1).repeat(1, num_samples, 1)
            expanded_mileage = batch['mileage'].unsqueeze(1).repeat(1, num_samples, 1)

            batched_sampled = topk_p_sampling(batch['input_ids'], prob_x, min_context, 
                                            k=params['sampling']['topk'], n=num_samples,
                                            p=params['sampling']['topp'],
                                            temp=params['sampling'].get('temp', None),
                                            cls_token_id=tokenizer.cls_token_id)
            with torch.inference_mode():
                o_ep = tfy(attention_mask=expanded_attention_mask.reshape(-1, batched_sampled.size(-1)), 
                            input_ids=batched_sampled.reshape(-1, batched_sampled.size(-1)), 
                            time=expanded_time.reshape(-1, batched_sampled.size(-1)), 
                            mileage=expanded_mileage.reshape(-1, batched_sampled.size(-1)),
                        )
                prob_y_sampled = o_ep['ep_prediction'].reshape(params['BS'], num_samples, batch['input_ids'].size(-1)-c, -1) # put back 
                prob_y_sampled = torch.clamp(prob_y_sampled, eps, 1 - eps)

            y_hat_i = prob_y_sampled[..., :-1, :] # P(Yi,j|z)
            y_hat_iplus1 = prob_y_sampled[..., 1:, :] # P(Yi+1,j|z, x)

            cmi = torch.mean(y_hat_iplus1*torch.log(y_hat_iplus1/y_hat_i)+ (1-y_hat_iplus1)*torch.log((1-y_hat_iplus1)/(1-y_hat_i)), dim=1)
            cs = torch.mean(y_hat_iplus1 - y_hat_i, dim=1)
            
            # Gather across all processes
            cmi = accelerator.gather(cmi)
            cs = accelerator.gather(cs)
            if accelerator.is_main_process:    
                # Process each batch in multiple parallelized threads
                for s_i in range(0, gathered_items['input_ids'].shape[0]):
                    #try:
                    input_sequence_decoded = tokenizer.decode(gathered_items['input_ids'][s_i, :])
                    #except:
                    #continue
                    already_computed_sampled = {}
                    causes = []
                    CMIS = {}
                    CMIS_sampled = {}
                    CS = {}
                    CS_sampled = {}
                    try:
                        labels = torch.nonzero(gathered_items['label'][s_i])
                    except:
                        print(f"Max samples reached at: {batch_index*params['BS']* accelerator.num_processes}")
                        continue

                    L = gathered_items['attention_mask'][s_i].sum()
                    # Could accelerate to do per token_idx
                    for y_c in labels:
                        y_c = y_c.item()
                        label_name = reversed_dict[y_c]
                        # useless 
                        if eval_o.is_input_ids_present_on_rule_for_label(input_sequence_decoded, label_name) is False:
                            continue
                        potential_causes = []
                        just_causes = []
                        CMIS[y_c] = []
                        CS[y_c] = []

                        # Consider samples where the y hat i is already at > 90% hat_y_i ? Something excited y_i in context c
                        if not ((params.get('consider_already_high_hat_y_i').get('value', True) == False) and (y_hat_i_0[s_i, c, y_c] > params.get('consider_already_high_hat_y_i').get('threshold', 0.9))):        
                            for token_idx in range(0, L-c-1):    
                                cmi_token_idx = cmi[s_i, token_idx, y_c]
                                cs_token_idx = cs[s_i, token_idx, y_c]
                                CS[y_c].append(cs_token_idx)
                                CMIS[y_c].append(cmi_token_idx)

                                if params['threshold']['type'] == 'static':
                                    if cmi_token_idx >= params['threshold']['value']:
                                        if (params['consider_prevalence'] == False) and cs_token_idx < 0:
                                            pass
                                        else:
                                            potential_causes.append({'i': token_idx+c, 'token': gathered_items['input_ids'][s_i, c+token_idx].item(), 
                                                       'y_i': y_hat_i[s_i, token_idx, y_c].item(), 'y_i+1': y_hat_iplus1[s_i, token_idx, y_c].item(), 'cs': cs_token_idx.item()})
                                            if params['VERBOSE']:
                                                print(potential_causes[-1])
                                            just_causes.append(gathered_items['input_ids'][s_i, token_idx+c+1].item())

                            # Dynamic Threshold 
                            try:
                                _ = torch.stack(CMIS[y_c], dim=0)
                                # condifence interval on CMI (mean - k.sigma)
                                mean_cmi = _.mean(dim=0)
                                std_cmi = _.std(dim=0)
                                dynamic_threshold = mean_cmi + params['threshold']['value'] * std_cmi
                                thresholds[y_c] = dynamic_threshold
                                for token_idx in range(0, L-c-1):
                                    if CMIS[y_c][token_idx] >= dynamic_threshold:
                                        if (params['consider_prevalence'] is False) and CS[y_c][token_idx] < 0:
                                            pass
                                        else:
                                            potential_causes.append({'i': token_idx+c, 'token': gathered_items['input_ids'][s_i, token_idx+c+1].item(), 
                                                               'cs': CS[y_c][token_idx].item()})
                                            if params['VERBOSE']:
                                                print(potential_causes[-1])
                                            just_causes.append(gathered_items['input_ids'][s_i, token_idx+c+1].item()) # this one will be use for evaluation
                            except:
                                print("Empty CMIS list, continuying ..")

                            params['eval_o'].eval_on_sample(input_sequence_decoded, y_c, label_name, just_causes)
                            causes.append({'label': y_c, 'causes': potential_causes, 'CMIS': CMIS[y_c]})
                        else:
                            print("Not considering already high y_i ..")

            total_step+=1
            # Break after a certain number of batches for demonstration (optional)
        if accelerator.is_main_process:
            return causes, thresholds
        else:
            return None, None
    
from accelerate import Accelerator
from accelerate.utils import gather_object

accelerator = Accelerator(mixed_precision='fp16')

# Debug the nb of GPU
message=[ f"Hello this is GPU {accelerator.process_index}" ] 

# collect the messages from all GPUs
messages=gather_object(message)

# output the messages only on the main process with accelerator.print() 
accelerator.print(messages)

params = {
    'BS': 12,
    'VERBOSE': False,
    'PRINT': False,
    'url_ep_def': "data/mb_labels.json",
    'context': 20,
    'threshold': {'type': 'z-score', 'value': 2.75},
    'consider_prevalence': True,
    'consider_not_present': False,
    'consider_already_high_hat_y_i': {'value': True, 'threshold': 0.9},
    'fp16': True,
    'eps': 1e-6,
    'multiple_gpus': True
}
tokpp_sampling = {'type': 'topk-p', 'value': 64, 'topk': 35, 'topp': 0.8, 'name': 'topp_35_0.8', 'temp': None}

def n_samples_comparaison():
    ##### N samples using 9 folds
    directory='comparaison/random_samples_n=500'
    RESULTS_FOLDER = "comparaison/random_samples_n=500"
    # Define the folder to store results
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    config = tokpp_sampling
    i = 1
    label_count = {}
    results = {}

    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Print the DataFrame or perform any operation
            ds_test = load_dataset(f"parquet", data_files=f"{directory}/{filename}")['train']
            print("\n")  # Add a newline for better readability
            print(f"from {filename}, with rows {len(ds_test)}")
            params['samples'] = len(ds_test) // (params['BS'])

            print(f"[!] Run {i}")
            start_time = datetime.now()
            params['sampling'] = config
            print(config)
            eval_o = EvaluationVehicleDataset(params['url_ep_def'], label_mapping, tokenizer, verbose=False, consider_prevalence=params['consider_prevalence'], consider_not_present=params['consider_not_present'])
            params['eval_o'] = eval_o
            causes, thresholds = OSCAR_parallel(params, ds_test, accelerator)
            i+=1

            if accelerator.is_main_process:
                print(f"\n\nRESULT\nMACRO AVG(all class treated equally) {eval_o.macro_average()}\nMICRO AVG(average across all samples){eval_o.micro_average()}\nWeighted AVG(weighted per label) {eval_o.weighted_average()}")
                end_time = datetime.now()
                # Calculate elapsed time
                elapsed_time = end_time - start_time
                print(f"Time spent: {elapsed_time.total_seconds():.4f} seconds")

                if results == {}:
                    print("Empty results")
                    results = {
                        'f1_score_micro': [eval_o.micro_average()['f1_score']], 
                        'precision_micro': [eval_o.micro_average()['precision']],
                        'recall_micro': [eval_o.micro_average()['recall']],

                        'f1_score_macro': [eval_o.macro_average()['f1_score']], 
                        'precision_macro': [eval_o.macro_average()['precision']],
                        'recall_macro': [eval_o.macro_average()['recall']],

                        'f1_score_weighted': [eval_o.weighted_average()['f1_score']],
                        'precision_weighted': [eval_o.weighted_average()['precision']],
                        'recall_weighted': [eval_o.weighted_average()['recall']],
                        'elapsed_time_sec': [elapsed_time.total_seconds()]
                    }
                else:
                    results['f1_score_micro'].append(eval_o.micro_average()['f1_score'])
                    results['precision_micro'].append(eval_o.micro_average()['precision'])
                    results['recall_micro'].append(eval_o.micro_average()['recall'])

                    results['f1_score_macro'].append(eval_o.macro_average()['f1_score'])
                    results['precision_macro'].append(eval_o.macro_average()['precision'])
                    results['recall_macro'].append(eval_o.macro_average()['recall'])

                    results['f1_score_weighted'].append(eval_o.weighted_average()['f1_score'])
                    results['precision_weighted'].append(eval_o.weighted_average()['precision'])
                    results['recall_weighted'].append(eval_o.weighted_average()['recall'])
                    results['elapsed_time_sec'].append(elapsed_time.total_seconds())
                print("RESULTS", results)

    result_file = os.path.join(RESULTS_FOLDER, "comparaison_across_samples.json")
    print("Saving to json ", result_file)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

##### Markov Len
def markov_len():
    print("Markov len comparaison")
    directory='dataset_markovlen'
    RESULTS_FOLDER='comparaison/markov_len_2'
    config = tokpp_sampling
    max_run = 5
    label_count = {}
    for filename in os.listdir(directory):
        results = {}
        if filename.endswith('.parquet'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)        
            # Print the DataFrame or perform any operation
            config_name = filename.replace('.parquet', '')
            s_test = load_dataset(f"parquet", data_files=f"{directory}/{filename}")['train']
            ds_test = s_test.map(tokenization, batched=True)
            print("\n")  # Add a newline for better readability
            print(f"from {filename}, with rows {len(ds_test)}")
            params['samples'] = len(ds_test) // (params['BS'])

            for i in range(0, max_run):
                print(f"[!] Run {i+1}/{max_run}")
                start_time = datetime.now()
                params['sampling'] = config
                print(config)
                eval_o = EvaluationVehicleDataset(params['url_ep_def'], label_mapping, tokenizer, verbose=False, consider_prevalence=params['consider_prevalence'], consider_not_present=params['consider_not_present'])
                params['eval_o'] = eval_o
                causes, thresholds = OSCAR_parallel(params, ds_test, accelerator)

                if accelerator.is_main_process:
                    print(f"\n\nRESULT\nMACRO AVG(all class treated equally) {eval_o.macro_average()}\nMICRO AVG(average across all samples){eval_o.micro_average()}\nWeighted AVG(weighted per label) {eval_o.weighted_average()}")
                    end_time = datetime.now()
                    # Calculate elapsed time
                    elapsed_time = end_time - start_time
                    print(f"Time spent: {elapsed_time.total_seconds():.4f} seconds")

                    if config_name not in results:
                        results[config_name] = {
                            'f1_score_micro': [eval_o.micro_average()['f1_score']], 
                            'precision_micro': [eval_o.micro_average()['precision']],
                            'recall_micro': [eval_o.micro_average()['recall']],

                            'f1_score_macro': [eval_o.macro_average()['f1_score']], 
                            'precision_macro': [eval_o.macro_average()['precision']],
                            'recall_macro': [eval_o.macro_average()['recall']],

                            'f1_score_weighted': [eval_o.weighted_average()['f1_score']],
                            'precision_weighted': [eval_o.weighted_average()['precision']],
                            'recall_weighted': [eval_o.weighted_average()['recall']],
                            'elapsed_time_sec': [elapsed_time.total_seconds()]
                        }
                    else:
                        results[config_name]['f1_score_micro'].append(eval_o.micro_average()['f1_score'])
                        results[config_name]['precision_micro'].append(eval_o.micro_average()['precision'])
                        results[config_name]['recall_micro'].append(eval_o.micro_average()['recall'])

                        results[config_name]['f1_score_macro'].append(eval_o.macro_average()['f1_score'])
                        results[config_name]['precision_macro'].append(eval_o.macro_average()['precision'])
                        results[config_name]['recall_macro'].append(eval_o.macro_average()['recall'])

                        results[config_name]['f1_score_weighted'].append(eval_o.weighted_average()['f1_score'])
                        results[config_name]['precision_weighted'].append(eval_o.weighted_average()['precision'])
                        results[config_name]['recall_weighted'].append(eval_o.weighted_average()['recall'])
                        results[config_name]['elapsed_time_sec'].append(elapsed_time.total_seconds())

                    result_file = os.path.join(RESULTS_FOLDER, f"{config_name}.json")
                    print("Saving to json ", result_file)
                    with open(result_file, "w") as f:
                        json.dump(results[config_name], f, indent=4)

n_samples_comparaison()
