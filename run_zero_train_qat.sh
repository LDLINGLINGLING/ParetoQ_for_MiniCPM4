#!/bin/bash

# 指定使用的GPU编号（如只用0,1,2,3号卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3
# =========================
# Zero 配置参数
# =========================
NUM_GPUS=4
MASTER_PORT=29500

# =========================
# 通用模型训练参数配置
# =========================
# 模型相关参数
export MODEL_INPUT_MODEL_FILENAME="/path/to/your/model"
export MODEL_OUTPUT_MODEL_LOCAL_PATH="/path/to/output/model.safetensors"
export MODEL_W_BITS=8
export MODEL_GROUP_SIZE=128
export MODEL_ONLY_TRAIN_ADAPTER="True"
export MODEL_USE_ORIGIN_MODEL="True"
export MODEL_ENTROPY_LOSS_WEIGHT=0.1

# 数据相关参数
export MODEL_TRAIN_DATA_LOCAL_PATH="/path/to/train_data.jsonl"
export MODEL_EVAL_DATA_LOCAL_PATH="/path/to/eval_data.jsonl"

# 训练相关参数
export MODEL_BF16="True"
export MODEL_MAX_LENGTH=256
export MODEL_OUTPUT_DIR="/path/to/output"
export MODEL_LOGGING_DIR="/path/to/logs"
export MODEL_PER_DEVICE_TRAIN_BATCH_SIZE=1
export MODEL_GRADIENT_ACCUMULATION_STEPS=8
export MODEL_NUM_TRAIN_EPOCHS=3
export MODEL_LEARNING_RATE=5e-6
export MODEL_WARMUP_STEPS=100
export MODEL_SAVE_STEPS=500
export MODEL_EVAL_STEPS=500
export MODEL_LM_LOSS_WEIGHT=1.0

# =========================
# 启动训练
# =========================
zero --num_gpus $NUM_GPUS --master_port $MASTER_PORT python train_qat.py
