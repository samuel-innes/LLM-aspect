#!/bin/bash

#SBATCH --job-name=ambig_binary_upsampled_a1
#SBATCH --output=output_files/upsampled/ambig_binary_upsampled_a1.txt
#SBATCH --ntasks=1 # assumably different if you want multiple tasks?
#SBATCH --time=72:00:00
#SBTACH --mail-user=innes@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 -u LLM-aspect/training_scripts/train_llama.py
# torchrun --nproc_per_node 1 --master_port 14549 TimeLlama/train_llama_chat.py\
#     --train_data_path LLM-aspect/data/preprocessed/umr/verb_class_train.json \
#     --eval_data_path LLM-aspect/data/preprocessed/umr/verb_class_eval.json \
#     --output_dir model7chat \
#     --num_train_epochs 70 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --tf32 False \
#     --bf16 False \
#     --gradient_accumulation_steps 4 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --gradient_checkpointing True \
#     --disable_tqdm False \
#     --learning_rate 5e-5  \
#     --fsdp "full_shard offload auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \