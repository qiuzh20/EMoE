while true; do
  # 获取 GPU 的数量
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

  # 使用 nvidia-smi 命令检查 GPU 状态，然后通过 awk 计算空闲 GPU 的数量
  idle_gpu_count=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,nounits,noheader | awk -F, '{if ($1/$2 > 0.9) print "idle"}' | wc -l)

  # 如果所有 GPU 都是空闲的，退出循环
  if [ "$idle_gpu_count" -eq "$gpu_count" ]; then
    echo "All GPUs are idle. Continuing with the task..."
    break
  else
    echo "Not all GPUs are idle. Waiting..."
    sleep 120  # 等待 60 秒再次检查
  fi
done


WORKER_GPU=8
WORKER_NUM=1


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
moe_train.py  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'MoELlamaDecoderLayer' \
    --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf-moe-64 \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_emoe_1104 \
    --split_start_layer 28 --split_every_layer 2 \
    --select 'gate'  --n_expert 64 \
    --topk 16


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
moe_train.py  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'MoELlamaDecoderLayer' \
    --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf-moe-64 \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_emoe_1104 \
    --split_start_layer 16 --split_every_layer 4 \
    --select 'gate'  --n_expert 64 \
    --topk 16