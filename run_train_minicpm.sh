#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "开始执行MiniCPM训练脚本..."

# 设置基础路径
BASE_DIR="/root/autodl-tmp"
PARETQ_DIR="${BASE_DIR}/ParetoQ-main"
MODEL_DIR="${BASE_DIR}/MiniCPM4-0_5B"
OUTPUT_DIR="${BASE_DIR}/output"
CACHE_DIR="${BASE_DIR}/cache"
DEEPSPEED_CONFIG="${BASE_DIR}/deepspeed_config.json"

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${BASE_DIR}/logs"

echo "1. 开始覆盖模型文件..."
# 强制覆盖models文件夹下的所有文件到模型目录
if [ -d "${PARETQ_DIR}/models" ]; then
    echo "正在将 ${PARETQ_DIR}/models/* 覆盖到 ${MODEL_DIR}/"
    cp -rf "${PARETQ_DIR}/models/"* "${MODEL_DIR}/"
    echo "模型文件覆盖完成"
else
    echo "警告: ${PARETQ_DIR}/models 目录不存在"
fi

echo "2. 开始训练..."
# 切换到ParetoQ-main目录
cd "${PARETQ_DIR}"

# 设置CUDA设备和分布式训练环境变量
export CUDA_VISIBLE_DEVICES=0
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 检查是否安装了deepspeed
if python -c "import deepspeed" 2>/dev/null; then
    echo "使用DeepSpeed进行训练..."
    # 使用deepspeed启动训练
    deepspeed --num_gpus=1 train_minicpm.py \
        --local_dir "${BASE_DIR}" \
        --input_model_filename "${MODEL_DIR}" \
        --output_model_filename "minicpm_quantized" \
        --train_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \
        --eval_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \
        --w_bits 4 \
        --contain_weight_clip_val False \
        --max_train_samples -1 \
        --max_eval_samples -1 \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_max_length 2048 \
        --do_train True \
        --do_eval True \
        --bf16 True \
        --logging_dir "${BASE_DIR}/logs" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --warmup_steps 100 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --eval_strategy "steps" \
        --save_strategy "steps" \
        --load_best_model_at_end True \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --remove_unused_columns False \
        --dataloader_pin_memory False \
        --optim "adamw_torch" \
        --deepspeed "${DEEPSPEED_CONFIG}"
else
    echo "DeepSpeed未安装，使用常规训练..."
    # 设置单GPU环境变量以避免分布式错误
    export RANK=0
    export LOCAL_RANK=0
    export WORLD_SIZE=1
    
    # 运行训练脚本（不使用deepspeed）
    python train_minicpm.py \
        --local_dir "${BASE_DIR}" \
        --input_model_filename "${MODEL_DIR}" \
        --output_model_filename "minicpm_quantized" \
        --train_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \
        --eval_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \
        --w_bits 4 \
        --contain_weight_clip_val False \
        --max_train_samples -1 \
        --max_eval_samples -1 \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_max_length 2048 \
        --do_train True \
        --do_eval True \
        --bf16 True \
        --logging_dir "${BASE_DIR}/logs" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --warmup_steps 100 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --eval_strategy "steps" \
        --save_strategy "steps" \
        --load_best_model_at_end True \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --remove_unused_columns False \
        --dataloader_pin_memory False \
        --optim "adamw_torch"
fi

echo "训练完成！"
echo "输出模型保存在: ${BASE_DIR}/models/minicpm_quantized"
echo "训练日志保存在: ${BASE_DIR}/logs"
echo "训练输出保存在: ${OUTPUT_DIR}"