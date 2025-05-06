# OSCAR- One Shot Causal AutoRegressive discovery

This repository is the official implementation of: *One-Shot Multi-Label Causal Discovery in High-Dimensional Event Sequences*. 

![oscar desc](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/Capture.PNG)
## Requirements

To install requirements:

```setup
pip install torch
```

Depending on your pretrained Transformer $\text{Tf}_x, \text{Tf}_y$ you might need additional packages.

## Settings & Pretraining

Two autoregressive transformers must be train on next event and label prediction before infering with OSCAR.
Such that you have a dataset of events (error codes, logs, symptoms: \$\boldsymbol{X}$) ordered with or w/o timestamps, and an associated outcome label(s) (disease, critical failure, defects: $\boldsymbol{Y}$) occuring at the end of the sequence of events.

The two models then output the conditionals: 

$P_{\theta_x}(X_i|\boldsymbol{Z})$ and 
$P_{\theta_y}(Y_j|X_i, \boldsymbol{Z})$ 
such that $X_i$ is the tested cause event and $Y_j$ is the effect label. 

## Inference

### Assumptions

It is important to note that OSCAR assume:
* *Temporal Precedence*: The sequence of events is corretly reccorded such that ordered event $x_i$ is allowed to influence any subsequence $x_j$ such that $t_i \leq t_j$ and $i<j$.
* *Bounded Lagged Effects*: Once we observed events up to a timestamp $t_i$, any future lagged copy of event $X^{t_i + \tau}_i$ does not additionally influence $Y_j$. In other words, we restrict the causal influence in a small interval once $X_i$ occurs. 
* *Causal Sufficiency*: All variables are observed
* *Oracle Models*: $\text{Tf}_x, \text{Tf}_y$ are trained perfectly such that they approximate the true distribution of the observed data.

### Batch inference
*Case: A batch of unobserved sequences need to be explain in term of causality between events and label occurence by an operator. Using previously trained autoregressive sequence models on the same doamin, OSCAR returns the indices of the events causing each labels for each sequence. Every operation is parallelized on GPUs. 
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

    # Unscramble to match original top-k indices
    # Need to reorder the sorted indices back to original top-k
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

        # Broadcast to select individual dynamic thresold
        cmi_mask = cmi >= dynamic_thresholds.unsqueeze(1)

        cause_token_indices = cmi_mask.nonzero(as_tuple=False)
        # (num_causes, 3) --> each row is [batch_idx, position_idx, label_idx]
        return cause_token_indices, cs_mean, cs_std, cmi_mask
```


## Evaluation

### Vehicular Event Dataset
OSCAR was evaluated on a test data of diagnostic trouble codes (as $X$) leading to failures of vehicles namely error pattern(s) (as $Y_j$). It is composed of about 8710 different diagnostic trouble codes and 268 error patterns. The dataset is characterized by a long-tail problem for the error pattern such that the labels are highly imbalanced.

We reused the two pretrained Transformers $\text{Tf}_x$: *CarFormer* and $\text{Tf}_y$: *EPredictor* from [Math et al.](https://arxiv.org/pdf/2412.13041) to perform the CI-tests on this dataset.
The evaluation of the different experiments are given under *eval.py*. 

### Results
The one-shot results on the Markov Boundary of each label (error pattern) are given here: 
| **Algorithm** | **Precision ↑**  | **Recall ↑**     | **F1 ↑**         | **Running Time (min) ↓** |
| ------------- | ---------------- | ---------------- | ---------------- | ------------------------ |
| IAMB          | -                | -                | -                | >1440                    |
| CMB           | -                | -                | -                | >1440                    |
| MB-By-MB      | -                | -                | -                | >1440                    |
| PCDbyPCD      | -                | -                | -                | >1440                    |
| **OSCAR**     | **39.49 ± 1.77** | **26.30 ± 0.89** | **29.01 ± 1.17** | **1.26**                 |

Standard multi-label causal discovery are not well adapted to high-dimensional event sequences, which they cannot solve in a reasonable amount of time.
Moreoever, it is easier to provide explaination for an operator per-sample on an unobserved data (one-shot) rather than solving the complete causal discovery problem across all the observational data (especially when having a lot of # event types and labels). Results might appear poor, however error patterns (labels) are imbalanced and getting refined over time by domain expert, making more difficult to extract the correct Markov Boundary, especially in a one-shot manner.

### Some graph exemples for error patterns on vehicles

![graph1](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/3Capture.PNG)
![graph2](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/Capture4.PNG)
![graph3](https://github.com/Mathugo/NeurIPS2025---OSCAR-One-Shot-Causal-AutoRegressive-Discovery/blob/main/21Capture.PNG)
