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

# 导入Transformers库的核心组件
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig
import copy
import torch
import transformers

# 导入自定义工具函数
from utils import utils
from utils import datautils

# 导入参数处理函数
from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer
from utils.trainer_ploss import CustomTrainerWithEntropyLoss
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

    # 配置模型相关参数（支持环境变量覆盖）
    model_args = Args()
    model_args.input_model_filename = os.environ.get("MODEL_INPUT_MODEL_FILENAME", "D:\model_best\minicpm\pretrain_model\Qwen3-0___6B")
    model_args.output_model_local_path = os.environ.get("MODEL_OUTPUT_MODEL_LOCAL_PATH", "D:\model_best\minicpm\pretrain_model\Qwen3-0___6B-GPTQ-Int8\model.safetensors")
    model_args.w_bits = int(os.environ.get("MODEL_W_BITS", 8))
    model_args.contain_weight_clip_val = os.environ.get("MODEL_CONTAIN_WEIGHT_CLIP_VAL", "False") == "True"
    model_args.group_size = int(os.environ.get("MODEL_GROUP_SIZE", 128))
    model_args.enable_groupwise = os.environ.get("MODEL_ENABLE_GROUPWISE", "True") == "True"
    model_args.gptq_model_path = os.environ.get("MODEL_GPTQ_MODEL_PATH", "D:\model_best\minicpm\pretrain_model\Qwen3-0___6B-GPTQ-Int8\model.safetensors")
    model_args.only_train_adapter = os.environ.get("MODEL_ONLY_TRAIN_ADAPTER", "True") == "True"
    model_args.use_origin_model = os.environ.get("MODEL_USE_ORIGIN_MODEL", "True") == "True"
    model_args.entropy_loss_weight = float(os.environ.get("MODEL_ENTROPY_LOSS_WEIGHT", 0.1))

    # 配置数据相关参数
    data_args = Args()
    data_args.train_data_local_path = os.environ.get("MODEL_TRAIN_DATA_LOCAL_PATH", r"D:\model_best\minicpm\ParetoQ_for_MiniCPM4\data\train_text.jsonl")
    data_args.eval_data_local_path = os.environ.get("MODEL_EVAL_DATA_LOCAL_PATH", r"D:\model_best\minicpm\ParetoQ_for_MiniCPM4\data\training_dataset_example.jsonl")

    # 配置训练相关参数
    training_args = Args()
    training_args.bf16 = os.environ.get("MODEL_BF16", "True") == "True"
    training_args.cache_dir = os.environ.get("MODEL_CACHE_DIR", "/root/autodl-tmp/cache")
    training_args.model_max_length = int(os.environ.get("MODEL_MAX_LENGTH", 256))
    training_args.do_train = os.environ.get("MODEL_DO_TRAIN", "True") == "True"
    training_args.do_eval = os.environ.get("MODEL_DO_EVAL", "True") == "True"
    training_args.output_dir = os.environ.get("MODEL_OUTPUT_DIR", "/root/autodl-tmp/output")
    training_args.logging_dir = os.environ.get("MODEL_LOGGING_DIR", "/root/autodl-tmp/logs")
    training_args.per_device_train_batch_size = int(os.environ.get("MODEL_PER_DEVICE_TRAIN_BATCH_SIZE", 1))
    training_args.per_device_eval_batch_size = int(os.environ.get("MODEL_PER_DEVICE_EVAL_BATCH_SIZE", 1))
    training_args.gradient_accumulation_steps = int(os.environ.get("MODEL_GRADIENT_ACCUMULATION_STEPS", 8))
    training_args.num_train_epochs = int(os.environ.get("MODEL_NUM_TRAIN_EPOCHS", 3))
    training_args.learning_rate = float(os.environ.get("MODEL_LEARNING_RATE", 5e-6))
    training_args.warmup_steps = int(os.environ.get("MODEL_WARMUP_STEPS", 100))
    training_args.logging_steps = int(os.environ.get("MODEL_LOGGING_STEPS", 1))
    training_args.save_steps = int(os.environ.get("MODEL_SAVE_STEPS", 500))
    training_args.eval_steps = int(os.environ.get("MODEL_EVAL_STEPS", 500))
    training_args.evaluation_strategy = os.environ.get("MODEL_EVALUATION_STRATEGY", "steps")
    training_args.save_strategy = os.environ.get("MODEL_SAVE_STRATEGY", "steps")
    training_args.load_best_model_at_end = os.environ.get("MODEL_LOAD_BEST_MODEL_AT_END", "True") == "True"
    training_args.metric_for_best_model = os.environ.get("MODEL_METRIC_FOR_BEST_MODEL", "eval_loss")
    training_args.greater_is_better = os.environ.get("MODEL_GREATER_IS_BETTER", "False") == "True"
    training_args.remove_unused_columns = os.environ.get("MODEL_REMOVE_UNUSED_COLUMNS", "False") == "True"
    training_args.dataloader_pin_memory = os.environ.get("MODEL_DATALOADER_PIN_MEMORY", "False") == "True"

    return model_args, data_args, training_args


def load_model_with_specific_modeling_file(model_path, use_origin_model, config, training_args, dtype):
    """
    使用指定的建模文件加载模型
    
    Args:
        model_path: 模型文件夹路径
        use_origin_model: True使用modeling_qwen3.py(QAT模型)，False使用modeling_backup.py(原始模型)
        config: 模型配置对象
        training_args: 训练参数
        dtype: 数据类型（torch.bfloat16或torch.float）
    
    Returns:
        model: 加载的模型对象
    """
    # 使用上下文管理器临时切换建模文件
    with switch_modeling_file(model_path, use_origin_model):
        # 从预训练模型加载因果语言模型
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,  # 模型路径
            config=config,  # 模型配置
            cache_dir=training_args.cache_dir,  # 缓存目录
            torch_dtype=dtype,  # 数据类型
            low_cpu_mem_usage=False,  # 不使用低CPU内存模式
            device_map='cpu',  # 设备映射：加载到CPU
            trust_remote_code=True,  # 信任远程代码
        )
    return model


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
    # 检查是否在调试模式下运行（没有命令行参数）
    is_debug_mode = len(sys.argv) == 1
    
    if is_debug_mode:
        # 调试模式：使用预设的默认参数
        print("Running in debug mode with default arguments...")
        model_args, data_args, training_args = setup_debug_args()
        # 注意：调试模式下通常不初始化分布式训练
        # dist.init_process_group(backend="nccl")
    else:
        # 正常模式：使用命令行参数并初始化分布式训练
        dist.init_process_group(backend="nccl")  # 初始化NCCL分布式后端
        model_args, data_args, training_args = process_args()  # 处理命令行参数

    log.info("Start to load models...")
    
    # 根据训练参数确定模型数据类型
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    # 从预训练模型加载配置
    config = AutoConfig.from_pretrained(model_args.input_model_filename)
    
    # 设置量化相关配置
    config.w_bits = model_args.w_bits  # 权重量化位数
    config.group_size = model_args.group_size  # 分组量化的组大小
    config.enable_groupwise = model_args.enable_groupwise  # 启用分组量化
    
    # 加载QAT（量化感知训练）模型
    log.info("Loading QAT model...")
    model = load_model_with_specific_modeling_file(
        model_args.input_model_filename, 
        use_origin_model=True,  # 使用QAT建模文件
        config=config, 
        training_args=training_args, 
        dtype=dtype
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
    
    # 将模型移动到GPU设备
    model.cuda()
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    # 加载分词器
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
        report_to=[],  # 禁用wandb/tensorboard等报告工具
    )
# 根据配置决定是否加载原始模型
    if model_args.use_origin_model:
        log.info("Loading original model...")
        origin_model = load_model_with_specific_modeling_file(
            model_args.input_model_filename, 
            use_origin_model=False,  # 使用原始建模文件
            config=config, 
            training_args=training_args, 
            dtype=dtype
        )
         # 使用带熵损失的自定义Trainer
        log.info("Using CustomTrainerWithEntropyLoss...")
        trainer = CustomTrainerWithEntropyLoss(
            model=model,
            origin_model=origin_model,
            entropy_loss_weight=getattr(model_args, 'entropy_loss_weight', 0.1),
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
    if training_args.do_train:
        train_result = trainer.train()  # 开始训练
        trainer.save_state()  # 保存训练状态（优化器、调度器等）
        # 安全保存模型
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    # 执行评估（如果启用评估模式）
    if training_args.do_eval:
        model.to("cuda")  # 确保模型在GPU上
        metrics = trainer.evaluate()  # 执行评估
        
        # 计算评估样本数量
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        
        try:
            # 计算困惑度（perplexity）= exp(loss)
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            # 处理数值溢出情况（损失值过大）
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        # 记录和保存评估指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 等待所有进程完成（仅在分布式训练模式下）
    if not is_debug_mode:
        torch.distributed.barrier()


@contextmanager
def switch_modeling_file(model_path, use_origin_model=True):
    """
    上下文管理器，用于临时切换建模文件
    
    这个函数通过临时替换modeling_qwen3.py文件来切换不同的模型实现：
    - QAT模型：使用原始的modeling_qwen3.py
    - 原始模型：使用modeling_backup.py替换modeling_qwen3.py
    
    Args:
        model_path: 模型文件夹路径
        use_origin_model: True使用modeling_qwen3.py(QAT模型)，False使用modeling_backup.py(原始模型)
    
    Yields:
        None: 在指定的建模文件配置下执行代码块
    """
    # 定义相关文件路径
    modeling_py_path = os.path.join(model_path, "modeling_qwen3.py")  # QAT建模文件
    backup_py_path = os.path.join(model_path, "modeling_qwen3_backup.py")  # 原始建模文件备份
    temp_py_path = os.path.join(model_path, "modeling_qwen3_temp.py")  # 临时备份文件
    
    # 检查必要文件是否存在
    if not os.path.exists(modeling_py_path):
        raise FileNotFoundError(f"QAT modeling file not found: {modeling_py_path}")
    if not os.path.exists(backup_py_path):
        raise FileNotFoundError(f"Backup modeling file not found: {backup_py_path}")
    
    # 备份当前的modeling_qwen3.py文件
    if os.path.exists(modeling_py_path):
        shutil.copy2(modeling_py_path, temp_py_path)
    
    try:
        if use_origin_model:
            # 使用QAT模型，保持modeling_qwen3.py不变
            log.info("Loading QAT model using modeling_qwen3.py")
        else:
            # 使用原始模型，用modeling_backup.py替换modeling_qwen3.py
            log.info("Loading original model using modeling_backup.py")
            shutil.copy2(backup_py_path, modeling_py_path)
        
        # 执行代码块
        yield
        
    finally:
        # 无论如何都要恢复原始的modeling_qwen3.py文件
        if os.path.exists(temp_py_path):
            shutil.move(temp_py_path, modeling_py_path)


if __name__ == "__main__":
    # 程序入口点：启动训练流程
    train()