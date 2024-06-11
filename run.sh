export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
cd ~/Data/Projects/fz_LLM/FastChat
torchrun --nproc_per_node=1 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path ~/.cache/modelscope/hub/shakechen/Llama-2-7b-chat-hf \
    --data_path /home/ubuntu/Data/Projects/fz_LLM/all.json \
    --bf16 True \
    --output_dir output_vicuna \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True