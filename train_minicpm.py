# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from utils.init_params import *
import math
import argparse
import sys
# 导入LLaMA模型配置类
from models.configuration_minicpm import MiniCPMConfig
# 导入量化版本的LLaMA模型

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,AutoConfig
import copy
import torch
import transformers
# 导入工具函数
from utils import utils
from utils import datautils

# 导入参数处理函数
from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer

# 获取日志记录器
log = utils.get_logger("clm")


def setup_debug_args():
    """
    设置调试用的默认参数，用于VSCode直接调试
    """
    # 创建命名空间对象来模拟参数
    class Args:
        pass
    
    # 模型参数
    model_args = Args()
    model_args.input_model_filename = "D:\model_best\minicpm\pretrain_model\MiniCPM4-0_5B"  # 修改为你的模型路径
    model_args.output_model_local_path = "D:\model_best\minicpm\pretrain_model\MiniCPM4-qatout"
    model_args.w_bits = 4  # 量化位数
    model_args.contain_weight_clip_val = False
    model_args.group_size = 32  # 分组量化大小
    model_args.enable_groupwise = True  # 启用分组量化
    
    # 数据参数
    data_args = Args()
    data_args.train_data_local_path = r"D:\model_best\minicpm\ParetoQ_for_MiniCPM4\train_text.jsonl"  # 修改为你的训练数据路径
    data_args.eval_data_local_path = r"D:\model_best\minicpm\ParetoQ_for_MiniCPM4\training_dataset_example.jsonl"   # 修改为你的验证数据路径
    
    # 训练参数
    training_args = Args()
    training_args.bf16 = True
    training_args.cache_dir = "/root/autodl-tmp/cache"
    training_args.model_max_length = 512
    training_args.do_train = True
    training_args.do_eval = True
    training_args.output_dir = "/root/autodl-tmp/output"
    training_args.logging_dir = "/root/autodl-tmp/logs"
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1
    training_args.gradient_accumulation_steps = 8
    training_args.num_train_epochs = 3
    training_args.learning_rate = 5e-5
    training_args.warmup_steps = 0
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
    
    return model_args, data_args, training_args


def fix_nan_in_model(model, verbose=True, inplace=True):
    """
    自动修复模型中包含NaN值的参数 - 仅修改NaN位置
         
    Args:
        model: 要修复的PyTorch模型
        verbose: 是否打印修复信息
        inplace: 是否原地修改模型，如果False则返回修复后的新模型
    
    Returns:
        model: 修复后的模型
        fixed_params: 修复的参数数量
        total_nan_count: 总共修复的NaN值数量
    """
    import torch
    import copy
    
    # 如果不是原地修改，创建模型的深拷贝
    if not inplace:
        model = copy.deepcopy(model)
    
    fixed_params = 0
    total_nan_count = 0
         
    for name, param in model.named_parameters():
        if param.is_meta:
            continue  # 跳过meta tensor
                     
        # 检查是否包含NaN
        nan_mask = torch.isnan(param)
        if not nan_mask.any():
            continue  # 没有NaN则跳过
        
        # 统计NaN数量
        nan_count = nan_mask.sum().item()
        total_nan_count += nan_count
                     
        # 获取设备和数据类型信息
        device = param.device
        dtype = param.dtype
        
        # 根据参数名决定替换策略
        with torch.no_grad():  # 确保不影响梯度计算
            if 'scale' in name.lower():
                # 整个scale参数矩阵替换为1
                param.data = torch.ones_like(param, device=device, dtype=dtype)
                if verbose:
                    print(f"Fully reset [scale] param {name} -> all 1.0 (shape {param.shape})")
                    
            elif 'zero' in name.lower() or 'bias' in name.lower():
                # 整个zero/bias参数矩阵替换为0
                param.data = torch.zeros_like(param, device=device, dtype=dtype)
                if verbose:
                    print(f"Fully reset [zero/bias] param {name} -> all 0.0 (shape {param.shape})")
                         
            elif 'weight' in name.lower():
                # weight参数的NaN位置用Xavier/Glorot初始化的值替换
                fan_in = param.size(-1) if param.dim() > 1 else param.numel()
                fan_out = param.size(0) if param.dim() > 1 else 1
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                replacement_values = torch.randn_like(param, device=device, dtype=dtype) * std
                param.data = torch.where(nan_mask, replacement_values, param.data)
                if verbose:
                    print(f"Fixed [weight] param {name}: {nan_count} NaN values -> Xavier init (shape {param.shape})")
            
            else:
                # 其他参数的NaN位置用小的随机正态分布值替换
                replacement_values = torch.randn_like(param, device=device, dtype=dtype) * 0.02
                param.data = torch.where(nan_mask, replacement_values, param.data)
                if verbose:
                    print(f"Fixed param {name}: {nan_count} NaN values -> small random (shape {param.shape})")
        
        fixed_params += 1

    if verbose:
        print(f"\nSummary:")
        print(f"- Fixed parameters: {fixed_params}")
        print(f"- Total NaN values fixed: {total_nan_count}")
        print(f"- Model modification: {'In-place' if inplace else 'New copy'}")
        
    return model, fixed_params, total_nan_count



def train():
    """
    主训练函数，负责模型训练和评估的完整流程
    """
    # 检查是否在调试模式下运行（没有命令行参数）
    is_debug_mode = len(sys.argv) == 1
    if is_debug_mode:
        # 调试模式：使用默认参数
        print("Running in debug mode with default arguments...")
        model_args, data_args, training_args = setup_debug_args()
        # 如果使用分布式训练，在调试时可能需要注释掉这行
        # dist.init_process_group(backend="nccl")
    else:
        # 正常模式：使用命令行参数
        dist.init_process_group(backend="nccl")
        model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    # 根据训练参数确定数据类型（bfloat16或float32）
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    # 从预训练模型加载配置
    config = MiniCPMConfig.from_pretrained(model_args.input_model_filename)
    # 设置权重量化位数
    config.w_bits = model_args.w_bits
    # 新增：设置分组量化配置
    config.group_size = model_args.group_size
    config.enable_groupwise = model_args.enable_groupwise
    # 加载量化版本的LLaMA模型
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,  # 使用低CPU内存模式
        device_map='cpu',        # 首先加载到CPU
        trust_remote_code=True,  # 信任远程代码
    )
    # has_nan, nan_info = check_nan_in_model(model, detailed=True)
    # if has_nan:
    #     model, fixed_params, total_fixed = fix_nan_in_model(model, verbose=True, inplace=True)
    model = initialize_quantization_params(model,group_size=model_args.group_size)
    # model = only_train_adapter(model, verbose=True)
    # 将模型移动到GPU
    model.cuda()
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    # 加载LLaMA分词器
    #tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
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
    # 创建自定义的JSON数据集对象用于训练
    train_data = datautils.CustomJsonDataset(
        train_dataset, tokenizer, block_size=training_args.model_max_length
    )
    # 创建验证数据集，限制最大长度为1024
    valid_data = datautils.CustomJsonDataset(
        valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
    )
    # 禁用模型缓存以节省内存
    model.config.use_cache = False
    # Create TrainingArguments from the args object
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
        eval_strategy=getattr(training_args, 'eval_strategy', getattr(training_args, 'evaluation_strategy', 'no')),
        save_strategy=training_args.save_strategy,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
        remove_unused_columns=training_args.remove_unused_columns,
        dataloader_pin_memory=training_args.dataloader_pin_memory,
        report_to=[],  # Disable wandb/tensorboard reporting by default
    )
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer for future compatibility
        args=hf_training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,  # 使用默认的数据整理器
    )

    # 执行训练（如果启用训练模式）
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()  # 保存训练状态
        # 安全保存模型
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    # 执行评估（如果启用评估模式）
    if training_args.do_eval:
        model.to("cuda")  # 确保模型在GPU上
        metrics = trainer.evaluate()  # 执行评估
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            # 计算困惑度（perplexity）
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            # 处理数值溢出情况
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        # 记录和保存评估指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 等待所有进程完成（分布式训练同步）
    if not is_debug_mode:  # Only use barrier in distributed mode
        torch.distributed.barrier()


if __name__ == "__main__":
    # 程序入口点
    train()
