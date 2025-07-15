#!/bin/bash

# =========================
# DeepSpeed 配置参数
# =========================
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
DEEPSPEED_CONFIG="/home/featurize/work/ParetoQ_for_MiniCPM4/deepspeed_config.json"

# 指定使用的GPU编号
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# =========================
# 模型训练参数配置（可通过命令行参数传递）
# =========================
# 模型相关参数
INPUT_MODEL_FILENAME=${1:-"/home/featurize/work/ParetoQ_for_MiniCPM4/models/Qwen3-0___6B"}
OUTPUT_MODEL_LOCAL_PATH=${2:-"/home/featurize/work/ParetoQ_for_MiniCPM4/output/model.safetensors"}
W_BITS=${3:-8}
GROUP_SIZE=${4:-128}
ONLY_TRAIN_ADAPTER=${5:-"True"}
USE_ORIGIN_MODEL=${6:-"True"}
ENTROPY_LOSS_WEIGHT=${7:-0.1}
LM_LOSS_WEIGHT=${8:-1.0}
MODEL_TYPE=${22:-"qwen3_qat"}
MODELING_FILENAME=${23:-"modeling_qwen3_qat.py"}

# 数据相关参数
TRAIN_DATA_LOCAL_PATH=${9:-"/home/featurize/work/ParetoQ_for_MiniCPM4/data/train_text.jsonl"}
EVAL_DATA_LOCAL_PATH=${10:-"/home/featurize/work/ParetoQ_for_MiniCPM4/data/training_dataset_example.jsonl"}

# 训练相关参数
BF16=${11:-"True"}
MAX_LENGTH=${12:-256}
OUTPUT_DIR=${13:-"/home/featurize/work/ParetoQ_for_MiniCPM4/output"}
PER_DEVICE_TRAIN_BATCH_SIZE=${14:-1}
GRADIENT_ACCUMULATION_STEPS=${15:-8}
NUM_TRAIN_EPOCHS=${16:-3}
LEARNING_RATE=${17:-5e-6}
WARMUP_STEPS=${18:-100}
SAVE_STEPS=${19:-500}
EVAL_STEPS=${20:-500}

# GPTQ模型路径（可选）
GPTQ_MODEL_PATH=${21:-"/home/featurize/work/ParetoQ_for_MiniCPM4/models/Qwen3-0___6B-GPTQ-Int8/model.safetensors"}

echo "==========================="
echo "启动DeepSpeed训练"
echo "==========================="
echo "模型输入路径: $INPUT_MODEL_FILENAME"
echo "模型输出路径: $OUTPUT_MODEL_LOCAL_PATH"
echo "训练数据路径: $TRAIN_DATA_LOCAL_PATH"
echo "验证数据路径: $EVAL_DATA_LOCAL_PATH"
echo "使用GPU数量: $NUM_GPUS"
echo "==========================="

# =========================
# 启动训练
# =========================
deepspeed --num_gpus $NUM_GPUS --master_port $MASTER_PORT train_qat.py \
    --input_model_filename "$INPUT_MODEL_FILENAME" \
    --output_model_local_path "$OUTPUT_MODEL_LOCAL_PATH" \
    --w_bits $W_BITS \
    --group_size $GROUP_SIZE \
    --only_train_adapter "$ONLY_TRAIN_ADAPTER" \
    --use_origin_model "$USE_ORIGIN_MODEL" \
    --entropy_loss_weight $ENTROPY_LOSS_WEIGHT \
    --lm_loss_weight $LM_LOSS_WEIGHT \
    --train_data_local_path "$TRAIN_DATA_LOCAL_PATH" \
    --eval_data_local_path "$EVAL_DATA_LOCAL_PATH" \
    --bf16 "$BF16" \
    --model_max_length $MAX_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --gptq_model_path "$GPTQ_MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --modeling_filename "$MODELING_FILENAME" \
    --deepspeed "$DEEPSPEED_CONFIG"
