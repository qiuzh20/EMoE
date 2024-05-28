# EMoE tuning for Llama

## Introduction

This is a repo for tuning EMoE for Llama. Since turely implementing MoE is time-consuming and causing inference latenc, we implement MoE through:
1. Rearrange weights of FFNs (see `moefication.py`).
2. Do block masking for FFNs during training, which is equivalent to MoE (see `moe_models.py`).
Actually this is faster than turely implementing MoE.

## Requirements

```shell
pip install k_means_constrained transformers==4.33.3  accelerate==0.23.0 tokenizers==0.13.3 datasets
```

## Preparation

```shell
python moefication.py --source_dir <path to original pretrained model> --output_dir <path rearranged model> --n_expert <n_expert>
# for Llama-7B, we recommend n_expert=64
# for Llama-14B, we recommend n_expert=128
```

## Training

```shell
bash moe_train.sh
```

related hyperparameters are in `moe_train.sh` and `moe_models.py`:

- model_name_or_path: the path to the rearranged pretrained model
- base_config_dir: the path to the base config file
- output_dir: the path to the output directory
- split_start_layer: the layer index to start splitting
  - suggested: 16 / 28 (last half or last quarter of the model)
- split_every_layer: the number of layers to split
  - suggested: 4 / 2 (split every 4 or 2 layers)
- select: use which intermediate results in FFNs to select experts ('gate', 'up', 'inter', 'inter_abs')
  - suggested: 'gate' (using gate output)
- n_expert: the number of experts
  - suggested: 64 / 128
- topk: the number of experts to select for each token
  - suggested: 16 / 32 / 64 (1/4 or 1/2 of n_expert)
