#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "开始执行MiniCPM训练脚本..."

# ============ 路径变量设置 ============
BASE_DIR="/root/autodl-tmp"                    # 设置项目根目录为/root/autodl-tmp
PARETQ_DIR="${BASE_DIR}/ParetoQ-main"          # 设置ParetoQ项目目录路径
MODEL_DIR="${BASE_DIR}/MiniCPM4-0_5B"         # 设置原始模型存放目录路径
OUTPUT_DIR="${BASE_DIR}/output"                # 设置训练输出结果保存目录
CACHE_DIR="${BASE_DIR}/cache"                  # 设置缓存文件存放目录
DEEPSPEED_CONFIG="${BASE_DIR}/deepspeed_config.json"  # 设置DeepSpeed配置文件路径

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"                       # 创建输出目录（如果不存在）
mkdir -p "${CACHE_DIR}"                        # 创建缓存目录（如果不存在）
mkdir -p "${BASE_DIR}/logs"                    # 创建日志目录（如果不存在）

echo "1. 开始覆盖模型文件..."
# 强制覆盖models文件夹下的所有文件到模型目录
if [ -d "${PARETQ_DIR}/models" ]; then
    echo "正在将 ${PARETQ_DIR}/models/* 覆盖到 ${MODEL_DIR}/"
    cp -rf "${PARETQ_DIR}/models/"* "${MODEL_DIR}/"    # 递归复制所有文件并强制覆盖
    echo "模型文件覆盖完成"
else
    echo "警告: ${PARETQ_DIR}/models 目录不存在"
fi

echo "2. 开始训练..."
cd "${PARETQ_DIR}"                             # 切换工作目录到ParetoQ项目目录

# ============ 环境变量设置 ============
export CUDA_VISIBLE_DEVICES=0                  # 设置只使用第0号GPU设备
export RANK=0                                  # 设置当前进程的全局排名为0（主进程）
export LOCAL_RANK=0                            # 设置当前进程的本地排名为0（本机主进程）
export WORLD_SIZE=1                            # 设置参与训练的总进程数为1（单进程训练）
export MASTER_ADDR=localhost                   # 设置主节点地址为本机
export MASTER_PORT=29500                       # 设置主节点通信端口为29500

# 检查是否安装了deepspeed
if python -c "import deepspeed" 2>/dev/null; then
    echo "使用DeepSpeed进行训练..."
    # 使用deepspeed启动训练
    deepspeed \                                 # 使用deepspeed命令启动分布式训练
        --num_gpus=1 \                         # 设置使用的GPU数量为1
        train_minicpm.py \                     # 指定要执行的训练脚本
        
        # ============ 基础路径参数 ============
        --local_dir "${BASE_DIR}" \                            # 设置本地工作根目录
        --input_model_filename "${MODEL_DIR}" \                # 设置输入模型的完整路径
        --output_model_filename "minicpm_quantized" \          # 设置输出模型的文件名（不含路径）
        
        # ============ 数据集路径参数 ============
        --train_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \  # 设置训练数据集文件的完整路径
        --eval_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \   # 设置验证数据集文件的完整路径
        
        # ============ 量化相关参数 ============
        --w_bits 4 \                                           # 设置权重量化位数为4位
        --contain_weight_clip_val False \                      # 设置不包含权重裁剪值（禁用权重裁剪）
        --group_size 128 \                                     # 设置分组量化的组大小为128
        --enable_groupwise True \                              # 启用分组量化
        
        # ============ 数据采样参数 ============
        --max_train_samples -1 \                               # 设置最大训练样本数为-1（使用全部训练数据）
        --max_eval_samples -1 \                                # 设置最大验证样本数为-1（使用全部验证数据）
        
        # ============ 存储目录参数 ============
        --cache_dir "${CACHE_DIR}" \                           # 设置数据和模型缓存目录
        --output_dir "${OUTPUT_DIR}" \                         # 设置训练输出和检查点保存目录
        --logging_dir "${BASE_DIR}/logs" \                     # 设置训练日志保存目录
        
        # ============ 模型配置参数 ============
        --model_max_length 2048 \                              # 设置模型最大输入序列长度为2048个token
        
        # ============ 训练阶段控制参数 ============
        --do_train True \                                       # 设置执行训练阶段
        --do_eval True \                                        # 设置执行验证阶段
        
        # ============ 数值精度参数 ============
        --bf16 True \                                           # 设置使用bfloat16混合精度训练
        
        # ============ 训练轮数参数 ============
        --num_train_epochs 3 \                                 # 设置训练总轮数为3轮
        
        # ============ 批次大小参数 ============
        --per_device_train_batch_size 1 \                      # 设置每个设备的训练批次大小为1
        --per_device_eval_batch_size 1 \                       # 设置每个设备的验证批次大小为1
        --gradient_accumulation_steps 8 \                      # 设置梯度累积步数为8（有效批次大小=1×8=8）
        
        # ============ 学习率参数 ============
        --learning_rate 5e-5 \                                 # 设置初始学习率为0.00005
        --warmup_steps 100 \                                   # 设置学习率预热步数为100步
        
        # ============ 日志记录参数 ============
        --logging_steps 10 \                                   # 设置每10步记录一次训练日志
        
        # ============ 模型保存参数 ============
        --save_steps 500 \                                     # 设置每500步保存一次模型检查点
        --save_strategy "steps" \                              # 设置按步数保存模型的策略
        
        # ============ 模型验证参数 ============
        --eval_steps 500 \                                     # 设置每500步进行一次模型验证
        --eval_strategy "steps" \                              # 设置按步数进行验证的策略
        
        # ============ 最佳模型选择参数 ============
        --load_best_model_at_end True \                        # 设置训练结束时加载最佳模型
        --metric_for_best_model "eval_loss" \                  # 设置以验证损失作为最佳模型的评判标准
        --greater_is_better False \                            # 设置评判指标越小越好（因为是loss）
        
        # ============ 数据处理参数 ============
        --remove_unused_columns False \                        # 设置不移除数据集中未使用的列
        --dataloader_pin_memory False \                        # 设置数据加载器不使用内存锁定（节省内存）
        
        # ============ 优化器参数 ============
        --optim "adamw_torch" \                                # 设置使用PyTorch版本的AdamW优化器
        
        # ============ DeepSpeed配置参数 ============
        --deepspeed "${DEEPSPEED_CONFIG}"                      # 设置DeepSpeed配置文件路径
else
    echo "DeepSpeed未安装，使用常规训练..."
    # 重新设置环境变量确保单GPU训练正常
    export RANK=0                              # 重新设置进程排名为0
    export LOCAL_RANK=0                        # 重新设置本地排名为0
    export WORLD_SIZE=1                        # 重新设置总进程数为1
    
    # 运行常规训练（不使用deepspeed）
    python train_minicpm.py \                  # 使用python直接执行训练脚本
        # 所有参数设置与DeepSpeed版本完全相同，除了没有--deepspeed参数
        --local_dir "${BASE_DIR}" \
        --input_model_filename "${MODEL_DIR}" \
        --output_model_filename "minicpm_quantized" \
        --train_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \
        --eval_data_local_path "${PARETQ_DIR}/training_dataset_example.jsonl" \
        --w_bits 4 \
        --contain_weight_clip_val False \
        --group_size 128 \
        --enable_groupwise True \
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
echo "输出模型保存在: ${BASE_DIR}/models/minicpm_quantized"    # 显示量化模型的最终保存位置
echo "训练日志保存在: ${BASE_DIR}/logs"                        # 显示训练日志的保存位置  
echo "训练输出保存在: ${OUTPUT_DIR}"                           # 显示检查点等训练输出的保存位置
