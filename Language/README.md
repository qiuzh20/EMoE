# EMoE for Language ID and OOD tasks

## Preparation

### Environment

```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd ./EMoE/tutel
pip3 install ./

pip3 install -r requirements.txt
```

### OOD data

Most of the data can be loaded directly trough `datasets`. For some addtional self-collected OOD tasks, please download them from [GLUE-X](https://github.com/YangLinyi/GLUE-X) and put them in `./dataset/datasets_self_collected`.

## Start Experiments

### Full Fine-tuning

Full Fine-tuning BERT-Large on CoLA, default `search_glue_no_trainer.py` would search learning rates in `[2e-5, 3e-5, 5e-5]` and repeat seeds `[0, 1, 2]`.

```sh
# vanilla fine-tuning
python search_glue_no_trainer.py --model_name_or_path bert-base-cased --task_name cola

# noisy tuning
python search_glue_no_trainer.py --model_name_or_path bert-base-cased --task_name cola  --noise_tuning 0.15

# GMoE
python search_glue_no_trainer.py --model_name_or_path bert-base-cased --task_name cola --to_MoE --gate_type cosine_top --num_experts 6 --top_k 1 --moe_layers 10 --expert_repeat 6

# EMoE
python search_glue_no_trainer.py --model_name_or_path bert-base-cased --task_name cola --to_MoE --gate_type top --one_score --key_gate  --num_experts 64 --top_k 16 --moe_layers 10

# EMoE-learn
python search_glue_no_trainer.py --model_name_or_path bert-base-cased --task_name cola --to_MoE --gate_type cosine_top  --num_experts 64 --top_k 16 --moe_layers 10
```

### Parametric Efficient Tuning

Parametric efficient tuning GPT2-XL on CoLA, default `search_glue_no_trainer_peft.py` would search learning rates in `[2e-4, 3e-4, 5e-4]` and repeat seeds `[0, 1, 2]`.

```sh
# vanilla LoRA
python search_glue_no_trainer_peft.py --use_fp16 --model_name_or_path gpt2-xl --per_device_train_batch_size 8  --gradient_accumulation_steps 2 --task_name cola  

# vanilla LoRA + Block
python search_glue_no_trainer_peft.py --use_fp16 --model_name_or_path gpt2-xl --per_device_train_batch_size 8  --gradient_accumulation_steps 2 --task_name cola --tune_moe_layers_only --moe_layers 44

# GMoE
python search_glue_no_trainer_peft.py --use_fp16 --model_name_or_path gpt2-xl --per_device_train_batch_size 8  --gradient_accumulation_steps 2 --task_name cola  --tune_moe_layers_only --moe_layers 44 --to_MoE --gate_type cosine_top --num_experts 8 --top_k 2 --expert_repeat 8

# EMoE
python search_glue_no_trainer_peft.py --use_fp16 --model_name_or_path gpt2-xl --per_device_train_batch_size 8  --gradient_accumulation_steps 2 --task_name cola  --tune_moe_layers_only --tune_gates_only --moe_layers 44 --to_MoE --one_score --key_gate --num_experts 64 --top_k 32

# EMoE-learn
python search_glue_no_trainer_peft.py --use_fp16 --model_name_or_path gpt2-xl --per_device_train_batch_size 8  --gradient_accumulation_steps 2 --task_name cola  --tune_moe_layers_only --tune_gates_only --moe_layers 44 --to_MoE --gate_type cosine_top --num_experts 64 --top_k 32
```

### OOD testing

After finish ID training, do OOD testing with

```sh
python test_glue_no_trainer.py --task_name ${OOD_task_name} --use_fp16  --model_name_or_path gpt2-xl --source_dir ${experiment_dir_with_all_checkpoints}
```

the OOD results will be recorded in the corresponding subfiles.
