# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import importlib.util
import sys
import os
from transformers import AutoModelForCausalLM, AutoConfig
import transformers
import logging
import os
from typing import Dict
import torch
from datetime import datetime


def load_custom_qwen3_model(
    model_path,
    modeling_filename="modeling_qwen3_qat.py",
    config_overrides=None,
    model_type="qwen3_local"
):
    """
    动态注册并加载本地自定义 Qwen3 模型。
    参数:
        model_path: 模型目录
        modeling_filename: 本地模型实现文件名
        config_overrides: dict, 需要覆盖的 config 属性
        model_type: 注册到transformers的model_type
    返回:
        model: 加载好的模型实例
    """
    # 1. 动态加载本地模型实现
    modeling_file = os.path.join(model_path, modeling_filename)
    spec = importlib.util.spec_from_file_location("modeling_qwen3_local", modeling_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modeling_qwen3_local"] = module
    spec.loader.exec_module(module)

    # 2. 修改 Config 的 model_type 及其它属性
    module.Qwen3Config.model_type = model_type
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(module.Qwen3Config, k, v)

    # 3. 绑定模型类的 config_class
    module.Qwen3ForCausalLM.config_class = module.Qwen3Config

    # 4. 注册到 transformers
    AutoConfig.register(model_type, module.Qwen3Config)
    AutoModelForCausalLM.register(model_type, module.Qwen3Config, module.Qwen3ForCausalLM)

    # 5. 禁用缓存
    transformers.dynamic_module_utils.get_cached_module_file.__globals__["_init_trust_remote_code"] = False

    # 6. 清除缓存（安全判断）
    get_cached_module_file = transformers.dynamic_module_utils.get_cached_module_file
    if hasattr(get_cached_module_file, "cache_clear"):
        get_cached_module_file.cache_clear()

    # 7. 加载配置
    config = AutoConfig.from_pretrained(model_path)
    config.model_type = model_type
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # 8. 加载模型
    from modeling_qwen3_local import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(model_path, config=config)
    return model

# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name):
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {}
        for key in state_dict.keys():
            if "teacher" in key:
                continue
            cpu_state_dict[key] = state_dict[key]
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_local_rank():
    if os.environ["LOCAL_RANK"]:
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()

def setup_project_logging():
    """
    设置项目日志文件保存功能
    在项目根目录下的log文件夹中创建日志文件
    
    Returns:
        log_file_path: 日志文件路径
    """
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, "log")
    
    # 创建log文件夹（如果不存在）
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_qat_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 设置根日志记录器级别和处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # 确保transformers相关的所有日志都被记录
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)
    transformers_logger.addHandler(file_handler)
    
    # 添加训练器日志记录器
    trainer_logger = logging.getLogger("transformers.trainer")
    trainer_logger.setLevel(logging.INFO)
    trainer_logger.addHandler(file_handler)
    
    # 添加我们自定义的日志记录器
    clm_logger = logging.getLogger("clm")
    clm_logger.setLevel(logging.INFO)
    clm_logger.addHandler(file_handler)
    
    print(f"Training logs will be saved to: {log_file_path}")
    return log_file_path