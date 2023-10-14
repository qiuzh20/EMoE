# Domainbed Experiments

## Preparation

```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd ./EMoE/tutel
pip3 install ./

pip3 install -r requirements.txt
```

## Datasets

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

## Start Experiments

### Training

Train a vanilla ViT-S model:

```sh
 python3 -m domainbed.scripts.train\
       --data_dir=/mnt/share/agiuser/qiuzihan/data\
       --algorithm GMOE\
       --dataset PACS\
	   --n_hparams 1\
       --n_trials 3 \
	   --hparams '{"vanilla_ViT":true, "vit_type":"small"}'\
	   --output_dir outputs-vanilla
```

Train a GMoE ViT-S model:

```sh
 python3 -m domainbed.scripts.train\
       --data_dir=${data_dir}\
       --algorithm GMOE\
       --dataset PACS\
	   --n_hparams 1\
       --n_trials 3 \
	   --hparams '{"vanilla_ViT":false, "vit_type":"small",
	   "num_experts":6, "topk":2, "num_inter":1}'\
	   --output_dir outputs-vanilla
```

Train a EMoE ViT-S model:

```sh
 python3 -m domainbed.scripts.train\
       --data_dir=${data_dir}\
       --algorithm GMOE\
       --dataset PACS\
	   --n_hparams 1\
       --n_trials 3 \
	   --hparams '{"vanilla_ViT":false, "vit_type":"small",
	   "MoE_from_ffn":true, "router":"top", "one_score_gate":true,
	   "num_experts":24, "topk":4, "num_inter":2}'\
	   --output_dir outputs-vanilla
```

### Evaluation

After training, go to the corresponding `output_dir` and run

```sh
python3 -m domainbed.scripts.collect_results  --input_dir=${output_dir}
```

### Hyper-params

We put hparams for each dataset into
```sh
./domainbed/hparams_registry.py
```

Basically, you just need to choose `--algorithm` and `--dataset`. The optimal hparams will be loaded accordingly. 

