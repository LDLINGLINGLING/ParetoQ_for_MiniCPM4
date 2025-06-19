#!/usr/bin/env python3
"""
将分组量化的checkpoint-3转换为GPTQ格式的脚本
"""

import torch
import json
import shutil
import os
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import numpy as np

def convert_checkpoint_to_gptq(
    checkpoint_path="/root/autodl-tmp/output/checkpoint-3",
    gptq_template_path="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-format",
    output_path="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-converted"
):
    """
    将分组量化的checkpoint转换为GPTQ格式
    
    Args:
        checkpoint_path: 输入的checkpoint路径
        gptq_template_path: GPTQ模板路径 
        output_path: 输出路径
    """
    
    print("=== 开始转换checkpoint到GPTQ格式 ===")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 1. 复制配置文件和其他必要文件
    print("1. 复制配置文件...")
    files_to_copy = [
        "config.json", "generation_config.json", "tokenizer.json", 
        "tokenizer_config.json", "added_tokens.json", "quantize_config.json",
        "configuration_minicpm.py", "modeling_minicpm.py"
    ]
    
    for file_name in files_to_copy:
        src_path = os.path.join(gptq_template_path, file_name)
        dst_path = os.path.join(output_path, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制 {file_name}")
    
    # 特殊处理tokenizer.model（如果存在）
    tokenizer_model_path = os.path.join(gptq_template_path, "tokenizer.model")
    if os.path.exists(tokenizer_model_path):
        shutil.copy2(tokenizer_model_path, os.path.join(output_path, "tokenizer.model"))
        print("  复制 tokenizer.model")
    
    # 2. 读取checkpoint配置
    print("2. 读取checkpoint配置...")
    with open(os.path.join(checkpoint_path, "config.json"), 'r') as f:
        checkpoint_config = json.load(f)
    
    w_bits = checkpoint_config.get('w_bits', 4)
    group_size = checkpoint_config.get('group_size', 128)
    enable_groupwise = checkpoint_config.get('enable_groupwise', True)
    
    print(f"  量化位数: {w_bits}")
    print(f"  分组大小: {group_size}")
    print(f"  分组量化: {enable_groupwise}")
    
    if not enable_groupwise:
        raise ValueError("只支持分组量化的checkpoint转换")
    
    # 3. 加载checkpoint权重
    print("3. 加载checkpoint权重...")
    checkpoint_tensors = {}
    with safe_open(os.path.join(checkpoint_path, "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            checkpoint_tensors[key] = f.get_tensor(key)
    
    print(f"  加载了 {len(checkpoint_tensors)} 个张量")
    
    # 4. 转换权重到GPTQ格式
    print("4. 转换权重到GPTQ格式...")
    gptq_tensors = {}
    
    # 找出所有需要量化的线性层
    quantized_layers = set()
    for key in checkpoint_tensors.keys():
        if 'weight_clip_val' in key:
            # 从 'model.layers.0.mlp.down_proj.weight_clip_val' 提取 'model.layers.0.mlp.down_proj'
            layer_name = key.replace('.weight_clip_val', '')
            quantized_layers.add(layer_name)
    
    print(f"  找到 {len(quantized_layers)} 个量化层")
    
    # 转换每个量化层
    converted_count = 0
    for layer_name in sorted(quantized_layers):
        weight_key = layer_name + '.weight'
        scale_key = layer_name + '.weight_clip_val'
        
        if weight_key not in checkpoint_tensors or scale_key not in checkpoint_tensors:
            print(f"  警告: 跳过层 {layer_name}，缺少权重或scale")
            continue
        
        weight = checkpoint_tensors[weight_key]
        scales = checkpoint_tensors[scale_key]
        
        print(f"  转换层: {layer_name}")
        print(f"    权重形状: {weight.shape}")
        print(f"    Scale形状: {scales.shape}")
        
        # 转换为GPTQ格式
        gptq_weight, gptq_scales, gptq_qzeros, gptq_g_idx = convert_weight_to_gptq(
            weight, scales, w_bits, group_size
        )
        
        # 保存GPTQ格式的张量
        gptq_tensors[layer_name + '.qweight'] = gptq_weight
        gptq_tensors[layer_name + '.scales'] = gptq_scales
        gptq_tensors[layer_name + '.qzeros'] = gptq_qzeros
        gptq_tensors[layer_name + '.g_idx'] = gptq_g_idx
        
        converted_count += 1
        
        print(f"    -> qweight: {gptq_weight.shape}")
        print(f"    -> scales: {gptq_scales.shape}")
        print(f"    -> qzeros: {gptq_qzeros.shape}")
        print(f"    -> g_idx: {gptq_g_idx.shape}")
    
    # 复制其他非量化参数（如embedding、layer norm等）
    print("5. 复制非量化参数...")
    for key, tensor in checkpoint_tensors.items():
        # 跳过已处理的权重和scale参数
        if any(layer_name in key for layer_name in quantized_layers):
            continue
        
        # 复制其他参数
        gptq_tensors[key] = tensor
        print(f"  复制: {key} {tensor.shape}")
    
    # 6. 保存GPTQ格式的权重文件
    print("6. 保存GPTQ格式权重...")
    output_safetensors_path = os.path.join(output_path, "model.safetensors")
    save_file(gptq_tensors, output_safetensors_path)
    
    print(f"  保存了 {len(gptq_tensors)} 个张量到 {output_safetensors_path}")
    print(f"  成功转换了 {converted_count} 个量化层")
    
    print("\n=== 转换完成 ===")
    print(f"GPTQ格式模型已保存到: {output_path}")


def convert_weight_to_gptq(weight, scales, w_bits, group_size):
    """
    将权重和scale转换为GPTQ格式
    
    Args:
        weight: 原始权重张量 [out_features, in_features]
        scales: 分组scale张量 [out_features, num_groups]
        w_bits: 量化位数
        group_size: 分组大小
    
    Returns:
        qweight, scales, qzeros, g_idx
    """
    out_features, in_features = weight.shape
    num_groups = scales.shape[1]
    
    # 验证分组配置
    expected_groups = (in_features + group_size - 1) // group_size
    if num_groups != expected_groups:
        raise ValueError(f"Scale分组数 {num_groups} 与预期 {expected_groups} 不匹配")
    
    # 1. 重新组织权重为分组形式
    # 将权重重塑为 [out_features, num_groups, group_size]
    padded_in_features = num_groups * group_size
    if in_features < padded_in_features:
        # 如果需要，填充权重
        padding = torch.zeros(out_features, padded_in_features - in_features, 
                            dtype=weight.dtype, device=weight.device)
        weight_padded = torch.cat([weight, padding], dim=1)
    else:
        weight_padded = weight[:, :padded_in_features]
    
    weight_grouped = weight_padded.view(out_features, num_groups, group_size)
    
    # 2. 使用现有的scales进行量化
    # scales的形状是 [out_features, num_groups]
    # 扩展scales以匹配权重分组
    scales_expanded = scales.unsqueeze(2)  # [out_features, num_groups, 1]
    
    # 量化权重
    max_q = 2 ** w_bits - 1
    weight_normalized = weight_grouped / (scales_expanded + 1e-8)
    weight_quantized = torch.clamp(torch.round(weight_normalized), 0, max_q)
    
    # 3. 转换为GPTQ存储格式
    # GPTQ将量化权重以特定方式打包
    
    # qweight: 打包的量化权重
    if w_bits == 4:
        # 4位量化：每个uint32存储8个4位值
        weight_quantized_int = weight_quantized.to(torch.int32)
        weight_flat = weight_quantized_int.view(out_features, -1)
        
        # 确保可以被8整除（每个uint32存储8个4位值）
        total_elements = weight_flat.shape[1]
        if total_elements % 8 != 0:
            padding_size = 8 - (total_elements % 8)
            padding = torch.zeros(out_features, padding_size, dtype=torch.int32, device=weight_flat.device)
            weight_flat = torch.cat([weight_flat, padding], dim=1)
        
        # 打包为uint32
        weight_packed = weight_flat.view(out_features, -1, 8)
        qweight = torch.zeros(out_features, weight_packed.shape[1], dtype=torch.int32, device=weight.device)
        
        for i in range(8):
            qweight += weight_packed[:, :, i] << (i * 4)
    
    elif w_bits == 3:
        # 3位量化的打包逻辑
        weight_quantized_int = weight_quantized.to(torch.int32)
        weight_flat = weight_quantized_int.view(out_features, -1)
        
        # 3位需要特殊处理，这里简化处理
        elements_per_int32 = 32 // w_bits  # 10个3位值
        total_elements = weight_flat.shape[1]
        if total_elements % elements_per_int32 != 0:
            padding_size = elements_per_int32 - (total_elements % elements_per_int32)
            padding = torch.zeros(out_features, padding_size, dtype=torch.int32, device=weight_flat.device)
            weight_flat = torch.cat([weight_flat, padding], dim=1)
        
        weight_packed = weight_flat.view(out_features, -1, elements_per_int32)
        qweight = torch.zeros(out_features, weight_packed.shape[1], dtype=torch.int32, device=weight.device)
        
        for i in range(elements_per_int32):
            qweight += weight_packed[:, :, i] << (i * w_bits)
    
    else:
        raise ValueError(f"不支持的量化位数: {w_bits}")
    
    # 4. 生成qzeros（零点）
    # 对于对称量化，零点通常是 2^(w_bits-1)
    qzeros = torch.full_like(scales, 2 ** (w_bits - 1), dtype=torch.int32)
    
    # 5. 生成g_idx（分组索引）
    # g_idx指示每个输入特征属于哪个组
    g_idx = torch.zeros(in_features, dtype=torch.int32, device=weight.device)
    for group_id in range(num_groups):
        start_idx = group_id * group_size
        end_idx = min(start_idx + group_size, in_features)
        g_idx[start_idx:end_idx] = group_id
    
    # 6. 调整scales格式
    # GPTQ的scales通常是 [out_features, num_groups]，直接使用原始scales
    gptq_scales = scales.clone()
    
    return qweight, gptq_scales, qzeros, g_idx


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将分组量化checkpoint转换为GPTQ格式")
    parser.add_argument("--checkpoint_path", 
                       default="/root/autodl-tmp/output/checkpoint-3",
                       help="输入checkpoint路径")
    parser.add_argument("--gptq_template_path",
                       default="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-format", 
                       help="GPTQ模板路径")
    parser.add_argument("--output_path",
                       default="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-converted",
                       help="输出路径")
    
    args = parser.parse_args()
    
    try:
        convert_checkpoint_to_gptq(
            checkpoint_path=args.checkpoint_path,
            gptq_template_path=args.gptq_template_path, 
            output_path=args.output_path
        )
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
