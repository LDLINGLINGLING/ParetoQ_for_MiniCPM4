# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入工具函数和参数初始化模块
from utils.init_params import *
import math
import argparse
import sys
import os
import shutil
from contextlib import contextmanager
import logging
import os
from utils.utils import setup_project_logging
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# 导入Transformers库的核心组件
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig
import torch

# 导入自定义工具函数
from utils import utils
from utils import datautils

# 导入参数处理函数
from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer
from utils.trainer_ploss import CustomTrainerWithEntropyLoss
from utils.utils import load_custom_qwen3_model 
# 获取日志记录器实例
log = utils.get_logger("clm")


def setup_debug_args():
    """
    设置调试用的默认参数配置
    用于在VSCode中直接调试，无需命令行参数

    Returns:
        model_args: 模型相关参数
        data_args: 数据相关参数  
        training_args: 训练相关参数
    """
    import os
    class Args:
        pass

    # 配置模型相关参数
    model_args = Args()
    model_args.model_path = "/home/featurize/work/Qwen3-0.6B"
    model_args.output_model_local_path = "/home/featurize/work/ParetoQ_for_MiniCPM4/output"
    model_args.w_bits = 4
    model_args.group_size = 128
    model_args.enable_groupwise = True
    model_args.gptq_model_path = "/home/featurize/work/Qwen3-0.6B-GPTQ-Int4"
    model_args.only_train_adapter = True
    model_args.use_origin_model = True
    model_args.entropy_loss_weight = 0.1
    model_args.lm_loss_weight = 1.0
    model_args.symmetric = False  # 是否使用对称量化
    model_args.model_type = "qwen3_qat"  # 模型类型
    model_args.modeling_filename = "modeling_qwen3_qat.py"  # 使用的建模文件名
    model_args.deepspeed_config = None  # 调试模式下不使用deepspeed

    # 配置数据相关参数
    data_args = Args()
    data_args.train_data_local_path = "/home/featurize/work/ParetoQ_for_MiniCPM4/data/train_text.jsonl"
    data_args.eval_data_local_path = "/home/featurize/work/ParetoQ_for_MiniCPM4/data/training_dataset_example.jsonl"

    # 配置训练相关参数
    training_args = Args()
    training_args.bf16 = True
    training_args.cache_dir = "/home/featurize/work/ParetoQ_for_MiniCPM4/cache"
    training_args.model_max_length = 256
    training_args.do_train = True
    training_args.do_eval = True
    training_args.output_dir = "/home/featurize/work/ParetoQ_for_MiniCPM4/output"
    # 设置日志目录到项目log文件夹
    project_root = os.path.dirname(os.path.abspath(__file__))
    training_args.logging_dir = os.path.join(project_root, "log", "tensorboard")
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1
    training_args.gradient_accumulation_steps = 8
    training_args.num_train_epochs = 3
    training_args.learning_rate = 5e-6
    training_args.warmup_steps = 100
    training_args.logging_steps = 1
    training_args.save_steps = 500
    training_args.eval_steps = 500
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better = False
    training_args.remove_unused_columns = False
    training_args.dataloader_pin_memory = False
    #training_args.deepspeed = "/home/featurize/work/ParetoQ_for_MiniCPM4/deepspeed_config.json"

    return model_args, data_args, training_args



def train():
    """
    主训练函数
    负责模型训练和评估的完整流程，包括：
    1. 参数配置和环境设置
    2. 模型和分词器加载
    3. 数据集准备
    4. 训练器配置
    5. 模型训练和评估
    """
    # 首先设置项目日志文件保存
    log_file_path = setup_project_logging()
    
    # 检查是否在调试模式下运行（没有命令行参数）
    is_debug_mode = len(sys.argv) == 1
    
    if is_debug_mode:
        # 调试模式：使用预设的默认参数
        print("Running in debug mode with default arguments...")
        log.info("Running in debug mode with default arguments...")
        model_args, data_args, training_args = setup_debug_args()
        # 注意：调试模式下通常不初始化分布式训练
        # dist.init_process_group(backend="nccl")
    else:
        # 正常模式：使用命令行参数并初始化分布式训练
        dist.init_process_group(backend="nccl")  # 初始化NCCL分布式后端
        model_args, data_args, training_args = process_args()  # 处理命令行参数

    log.info("Start to load models...")
    log.info(f"Log file saved to: {log_file_path}")
    
    # 根据训练参数确定模型数据类型
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    # 从预训练模型加载配置
    config = AutoConfig.from_pretrained(model_args.model_path)
    
    # 设置量化相关配置
    config.w_bits = model_args.w_bits  # 权重量化位数
    config.group_size = model_args.group_size  # 分组量化的组大小
    config.enable_groupwise = model_args.enable_groupwise  # 启用分组量化
    config.symmetric = model_args.symmetric  # 是否使用对称量化
    

    # 加载QAT（量化感知训练）模型
    log.info("Loading QAT model...")
    config_qat = {
        "w_bits": config.w_bits,
        "group_size": config.group_size,
        "enable_groupwise": model_args.enable_groupwise,
        "symmetric": model_args.symmetric
    }
    model = load_custom_qwen3_model(
        model_path=model_args.model_path,
        modeling_filename=model_args.modeling_filename,
        config_overrides=config_qat,
        model_type="qwen3_local"
    )
        
    
    # 如果指定了GPTQ模型路径，加载GPTQ权重
    if model_args.gptq_model_path is not None:
        log.info(f"Loading GPTQ weights from {model_args.gptq_model_path}...")
        # 仅加载GPTQ权重（这些函数似乎在其他地方定义）
        gptq_weights = load_gptq_weights_only(model_args.gptq_model_path)
        model = initialize_from_gptq_model(model, gptq_weights)
        # 删除权重数据以释放内存
        del gptq_weights
        import gc; gc.collect()  # 强制垃圾回收
    
    # 如果只训练适配器，配置模型为适配器训练模式
    if model_args.only_train_adapter:
        model = only_train_adapter(model, verbose=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found in the model. Please check only_train_adapter or parameter freezing logic.")
    
    # 将模型移动到GPU设备
    model.cuda()
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",    # 右侧填充
        add_bos_token=False,     # 不添加句首标记
        add_eos_token=False,     # 不添加句尾标记
    )
    log.info("Complete tokenizer loading...")

    # 获取训练和验证数据集
    train_dataset, valid_dataset = datautils.get_train_val_dataset(
        train_path=data_args.train_data_local_path,
        valid_path=data_args.eval_data_local_path
        if data_args.eval_data_local_path is not None
        else None,
    )
    
    # 创建自定义JSON数据集对象用于训练
    train_data = datautils.CustomJsonDataset(
        train_dataset, tokenizer, block_size=training_args.model_max_length
    )
    
    # 创建验证数据集，限制最大长度为1024（防止显存不足）
    valid_data = datautils.CustomJsonDataset(
        valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
    )
    
    # 禁用模型缓存以节省显存
    model.config.use_cache = False
    
    # 从自定义参数对象创建HuggingFace TrainingArguments对象
    # 确保logging_dir目录存在
    os.makedirs(training_args.logging_dir, exist_ok=True)
    
    # 添加DeepSpeed配置到HuggingFace TrainingArguments
    deepspeed_config = getattr(model_args, 'deepspeed_config', None)
    
    hf_training_args = TrainingArguments(
        bf16=training_args.bf16,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        output_dir=training_args.output_dir,
        logging_dir=training_args.logging_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        warmup_steps=training_args.warmup_steps,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        # 兼容不同版本的参数名称
        eval_strategy=getattr(training_args, 'eval_strategy', getattr(training_args, 'evaluation_strategy', 'no')),
        save_strategy=training_args.save_strategy,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
        remove_unused_columns=training_args.remove_unused_columns,
        dataloader_pin_memory=training_args.dataloader_pin_memory,
        deepspeed=deepspeed_config,  # 添加DeepSpeed配置
        report_to=[],  # 禁用wandb/tensorboard等报告工具
    )

    skip_eval = False
    if deepspeed_config:
        with open(deepspeed_config, "r") as f:
            ds_cfg = json.load(f)
        zero_stage = ds_cfg.get("zero_optimization", {}).get("stage", 0)
        if zero_stage == 2:
            log.info("Detected DeepSpeed ZeRO Stage 2, skipping evaluation.")
            skip_eval = True

# 根据配置决定是否加载原始模型
    if model_args.use_origin_model:
        log.info("Loading original model...")
        origin_model =  AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_path,  # 模型路径
            config=config,  # 模型配置
            torch_dtype=dtype,  # 数据类型
            low_cpu_mem_usage=False,  # 不使用低CPU内存模式
            trust_remote_code=True,  # 信任远程代码
        )
         # 使用带熵损失的自定义Trainer
        log.info("Using CustomTrainerWithEntropyLoss...")
        trainer = CustomTrainerWithEntropyLoss(
            model=model,
            origin_model=origin_model,
            entropy_loss_weight=getattr(model_args, 'entropy_loss_weight', 0.1),
            lm_loss_weight=getattr(model_args, 'lm_loss_weight', 1.0),
            processing_class=tokenizer,
            args=hf_training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=valid_data if training_args.do_eval else None,
            data_collator=default_data_collator,
        )
    else:
        # 创建Trainer对象
        trainer = Trainer(
            model=model,
            processing_class=tokenizer,  # 使用processing_class而不是tokenizer以保证未来兼容性
            args=hf_training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=valid_data if training_args.do_eval else None,
            data_collator=default_data_collator,  # 使用默认的数据整理器
        )

    # 执行训练（如果启用训练模式）
    # if training_args.do_train:
    #     log.info("Starting training...")
    #     train_result = trainer.train()  # 开始训练
    #     log.info("Training completed")
    #     trainer.save_state()  # 保存训练状态（优化器、调度器等）
    #     # 安全保存模型
    #     log.info(f"Saving model to: {model_args.output_model_local_path}")
    #     utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)
    #     log.info("Model saved successfully")

    if training_args.do_eval and not skip_eval:
        log.info("Starting evaluation...")
        model.to("cuda")  # 确保模型在GPU上
        metrics = trainer.evaluate()  # 执行评估
        
        # 计算评估样本数量
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        
        try:
            # 计算困惑度（perplexity）= exp(loss)
            if "train_lm_loss" in metrics:
                perplexity = math.exp(metrics["train_lm_loss"])
            else:
                perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            # 处理数值溢出情况（损失值过大）
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        # 记录和保存评估指标
        log.info(f"Evaluation completed. Metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 等待所有进程完成（仅在分布式训练模式下）
    if not is_debug_mode:
        torch.distributed.barrier()
    
    log.info("Training script completed successfully")


if __name__ == "__main__":
    # 程序入口点：启动训练流程
    train()