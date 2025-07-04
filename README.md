# OSCAR- One Shot Causal AutoRegressive discovery

This repository contains the official implementation of the paper "One-Shot Multi-Label Causal Discovery in High-Dimensional Event Sequences", which introduces OSCAR: a novel approach for rapid causal discovery in high-dimensional, multi-label event sequences.

![oscar desc](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/imgs/Capture.PNG)

## Why Use OSCAR?
Causal discovery in event sequences is challenging due to the high dimensionality and complex temporal dependencies in real-world data. OSCAR addresses this by leveraging the expressive power of pretrained autoregressive models, making it uniquely suited for applications like:

Industrial fault diagnostics

Medical event analysis (e.g., disease progression)

Complex system monitoring (e.g., network logs, cyberattack detection)

## Requirements

To install requirements:

```setup
pip install torch transformers accelerate datasets numpy
```

Depending on your pretrained Transformer $\text{Tf}_x, \text{Tf}_y$ you might need additional packages.

## Settings & Pretraining

### Reuse the data from the paper

If you want to reuse the data, a parquet files containing 46,000 sequences of error codes are given under: *data/ds_test.parquet*
Which contains already tokenised and encoded events and labels. The ground truth Markov Boundary for each label (0 to 268) is given in 
*data/mb_labels.json*.

### Preparing Your Data 

Before using OSCAR, you need to train two autoregressive models:
1. Next Event Model ($\text{Tf}_x$): Predicts the next event in a sequence.
2. Next Label Model ($\text{Tf}_y$): Predicts the outcome labels given the current event and history.

Your training data should consist of sequences of events () and associated labels (), which may include:

* Events: Error codes, system logs, clinical symptoms
* Labels: Critical failures, disease diagnoses, system malfunctions

Ensure that your data is properly tokenised and prepared as required by your chosen transformer model.

### Training the models
You should first pretrain these models to estimate the following conditionals:

$P_{\theta_x}(X_i|\boldsymbol{Z})$ and 
$P_{\theta_y}(Y_j|X_i, \boldsymbol{Z})$ 

where $X_i$ is the candidate cause event and $Y_j$ is the effect label and $\boldsymbol{Z}$ is the observed event history$.

## Inference

### Assumptions

It is important to note that OSCAR assume:
* *Temporal Precedence*: The sequence of events is correctly recorded such that ordered event $x_i$ is allowed to influence any subsequence $x_j$ such that $t_i \leq t_j$ and $i<j$.
* *Bounded Lagged Effects*: Once we observed events up to a timestamp $t_i$, any future lagged copy of event $X^{t_i + \tau}_i$ does not additionally influence $Y_j$. In other words, we restrict the causal influence in a small interval once $X_i$ occurs. 
* *Causal Sufficiency*: All variables are observed
* *Oracle Models*: $\text{Tf}_x, \text{Tf}_y$ are trained perfectly such that they approximate the true distribution of the observed data.

### Batch inference
*Case: A batch of unobserved sequences needs to be explained in terms of causality between events and label occurrence by an operator. Using previously trained autoregressive sequence models on the same domain, OSCAR returns the indices of the events causing each label for each sequence. Every operation is parallelised on GPUs. 
For the full description of the parameters, please see the paper*.

```python
from torch import nn

def topk_p_sampling(z, prob_x, c: int, n: int = 64, p: float = 0.8, k: int = 20,
                       cls_token_id: int = 1, temp: float = None):
    # Sample just the context
    input_ = prob_x[:, :c]

    # Top-k first
    topk_values, topk_indices = torch.topk(input_, k=k, dim=-1)

    # Top-p over top-k values
    sorted_probs, sorted_idx = torch.sort(topk_values, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > p
    
    # Ensure at least one token is kept
    mask[..., 0] = 0

    # Mask and normalize
    filtered_probs = sorted_probs.masked_fill(mask, 0.0)
    filtered_probs += 1e-8  # for numerical stability
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

    # Unscramble to match the original top-k indices
    # Need to reorder the sorted indices back to the original top-k
    reorder_idx = torch.argsort(sorted_idx, dim=-1)
    filtered_probs = torch.gather(filtered_probs, -1, reorder_idx)

    batched_probs = filtered_probs.unsqueeze(1).repeat(1, n, 1, 1)        # (bs, n, seq_len, k)
    batched_indices = topk_indices.unsqueeze(1).repeat(1, n, 1, 1)        # (bs, n, seq_len, k)

    sampled_idx = torch.multinomial(batched_probs.view(-1, k), 1)         # (bs*n*seq_len, 1)
    sampled_idx = sampled_idx.view(-1, n, c).unsqueeze(-1)

    sampled_tokens = torch.gather(batched_indices, -1, sampled_idx).squeeze(-1)
    sampled_tokens[..., 0] = cls_token_id

    # Reconstruct full sequence
    z_expanded = z.unsqueeze(1).repeat(1, n, 1)[..., c:]
    return torch.cat((sampled_tokens, z_expanded), dim=-1)

```

```python

def OSCAR(tfe: nn.Module, tfy: nn.Module, batch: dict[str, torch.Tensor], c: int, n: int, eps: float=1e-6, topk: int=20, k: int=2.75, p=0.8) -> torch.Tensor:
    """ tfe, tfy: are the two autoregressive transformers (event type and label)
        batch: dictionary containing a batch of input_ids and attention_mask of shape (bs, L) to explain.
        c: scalar number defining the minimum context to start infering, also the sampling interval.
        n: scalar number representing the number of samples for the sampling method.
        eps: float for numerical stability
        topk: The number of top-k most probable tokens to keep for sampling
        k: Number of standard deviations to add to the mean for dynamic threshold calculation
        p: Probability mass for top-p nucleus
    """
    o = tfe(attention_mask=batch['attention_mask'], input_ids=batch['input_ids'])['prediction_logits'] # Infer the next event type
    x_hat = torch.nn.functional.softmax(o, dim=-1)

    b_sampled = topk_p_sampling(batch['input_ids'], x_hat, c, k=topk, n=n, p=p) # Sampling up to (bs, n, L)
    n_att_mask = batch['attention_mask'].unsqueeze(1).repeat(1, n, 1)

    with torch.inference_mode():
        o = tfy(attention_mask=n_att_mask.reshape(-1, b_sampled.size(-1)), input_ids=b_sampled.reshape(-1, b_sampled.size(-1))) # flatten and infer
        prob_y_sampled = o['ep_prediction'].reshape(b_sampled.size(0), n, batch['input_ids'].size(-1)-c, -1) # reshape to (bs, n, L-c)

        # Ensure probs are within (eps, 1-eps)
        prob_y_sampled = torch.clamp(prob_y_sampled, eps, 1 - eps)

        y_hat_i = prob_y_sampled[..., :-1, :] # P(Yj|z)
        y_hat_iplus1 = prob_y_sampled[..., 1:, :] # P(Yj|z, x_i) 

        # Compute the CMI & CS and Average across sampling dim
        cmi = torch.mean(y_hat_iplus1*torch.log(y_hat_iplus1/y_hat_i)+ (1-y_hat_iplus1)*torch.log((1-y_hat_iplus1)/(1-y_hat_i)), dim=1)
        # (BS, L, Y)
        cs = y_hat_iplus1 - y_hat_i
        cs_mean = torch.mean(cs, dim=1)
        cs_std = torch.std(cs, dim=1)

        # Confidence interval for threshold
        mu = cmi.mean(dim=1)
        std = cmi.std(dim=1)
        dynamic_thresholds = mu + std * k

        # Broadcast to select individual dynamic threshold
        cmi_mask = cmi >= dynamic_thresholds.unsqueeze(1)

        cause_token_indices = cmi_mask.nonzero(as_tuple=False)
        # (num_causes, 3) --> each row is [batch_idx, position_idx, label_idx]
        return cause_token_indices, cs_mean, cs_std, cmi_mask
```


## Evaluation

### Vehicular Event Dataset
OSCAR was evaluated on a test dataset of diagnostic trouble codes (as $X$) leading to failures of vehicles, namely error pattern(s) (as $Y_j$). It is composed of about 8710 different diagnostic trouble codes and 268 error patterns. The dataset is characterised by a long-tail problem for the error pattern, such that the labels are highly imbalanced.

We reused the two pretrained Transformers $\text{Tf}_x$: *CarFormer* and $\text{Tf}_y$: *EPredictor* [[1]](https://arxiv.org/pdf/2412.13041) to perform the CI-tests on this dataset.
The evaluation of the different experiments is given under *eval.py*. 

### Results
The one-shot results on the Markov Boundary of each label (error pattern) are given here: 
| **Algorithm** | **Precision ↑**  | **Recall ↑**     | **F1 ↑**         | **Running Time (min) ↓** |
| ------------- | ---------------- | ---------------- | ---------------- | ------------------------ |
| IAMB          | -                | -                | -                | >1440                    |
| CMB           | -                | -                | -                | >1440                    |
| MB-By-MB      | -                | -                | -                | >1440                    |
| PCDbyPCD      | -                | -                | -                | >1440                    |
| MI-MCF        | -                | -                | -                | >1440                    |
| **OSCAR**     | **39.49 ± 1.77** | **26.30 ± 0.89** | **29.01 ± 1.17** | **1.26**                 |

Standard multi-label causal discovery methods are not well adapted to high-dimensional event sequences, as they cannot solve in a reasonable amount of time.
Moreover, it is easier to provide explanation for an operator per-sample on unobserved data (one-shot) rather than solving the complete causal discovery problem across all the observational data (especially when having a lot of # event types and labels). Results might appear poor, however, error patterns (labels) are imbalanced and getting refined over time by a domain expert, making it more difficult to extract the correct Markov Boundary, especially in a one-shot manner.

### Graph examples for vehicle diagnostics

![graph1](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/imgs/3Capture.PNG)
![graph2](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/imgs/Capture4.PNG)
![graph3](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/imgs/21Capture.PNG)

## Reproducibility 

The file *comparaison_multigpus.py* contains the parallelised implementation and evaluation of OSCAR. 
Launch comparaisons with 4 gpus:

```ssh
accelerate launch --multi_gpu --num_processes=4 comparaison_multigpu.py 
```


## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. You may copy, distribute, remix, and build upon the material for non-commercial purposes only.
