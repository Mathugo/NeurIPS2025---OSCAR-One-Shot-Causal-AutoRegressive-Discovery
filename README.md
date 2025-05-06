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

### Batch exemple


## Evaluation

### Vehicular Event Dataset
OSCAR was evaluated on a test data of diagnostic trouble codes (as $X$) leading to failures of vehicles namely error pattern(s) (as $Y_j$). It is composed of about 8710 different diagnostic trouble codes and 268 error patterns. The dataset is characterized by a long-tail problem for the error pattern such that the labels are highly imbalanced.

We reused the two pretrained Transformers $\text{Tf}_x$: *CarFormer* and $\text{Tf}_y$: *EPredictor* from [Math et al.](https://arxiv.org/pdf/2412.13041) to perform the CI-test.

The evaluation of the different experiments are given under *eval.py*. 


### Results

| **Algorithm** | **Precision ↑**  | **Recall ↑**     | **F1 ↑**         | **Running Time (min) ↓** |
| ------------- | ---------------- | ---------------- | ---------------- | ------------------------ |
| IAMB          | -                | -                | -                | >1440                    |
| CMB           | -                | -                | -                | >1440                    |
| MB-By-MB      | -                | -                | -                | >1440                    |
| PCDbyPCD      | -                | -                | -                | >1440                    |
| **OSCAR**     | **39.49 ± 1.77** | **26.30 ± 0.89** | **29.01 ± 1.17** | **1.26**                 |


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
