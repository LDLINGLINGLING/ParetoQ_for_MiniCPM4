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

def convert_weight_to_gptq_simple(weight, scales, w_bits, group_size):
    """
    简化的权重转换方法，直接使用训练时的量化参数
    
    Args:
        weight: 原始浮点权重张量 [out_features, in_features]
        scales: 训练时的量化scale [out_features, num_groups]
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
    
    # 1. 使用与训练时相同的量化方式
    # 填充权重到完整分组
    padded_in_features = num_groups * group_size
    if in_features < padded_in_features:
        padding = torch.zeros(out_features, padded_in_features - in_features, 
                            dtype=weight.dtype, device=weight.device)
        weight_padded = torch.cat([weight, padding], dim=1)
    else:
        weight_padded = weight[:, :padded_in_features]
    
    # 重新组织为分组形式 [out_features, num_groups, group_size]
    weight_grouped = weight_padded.view(out_features, num_groups, group_size)
    scales_expanded = scales.unsqueeze(2)  # [out_features, num_groups, 1]
    
    # 2. 使用LSQ量化公式（与训练时一致）
    if w_bits == 1 or w_bits == 0:
        Qn, Qp = -1, 1
    else:
        Qn = -(2 ** (w_bits - 1))
        Qp = 2 ** (w_bits - 1) - 1
    
    # 量化：w_q = (w / scale).round().clamp(Qn, Qp)
    quantized_lsq = torch.clamp(
        torch.round(weight_grouped / scales_expanded), 
        Qn, Qp
    )
    
    # 3. 转换为GPTQ兼容格式：映射到 [0, 2^w_bits-1]
    max_q = 2 ** w_bits - 1
    quantized_gptq = quantized_lsq - Qn  # 将 [Qn, Qp] 映射到 [0, max_q]
    quantized_gptq = torch.clamp(quantized_gptq, 0, max_q).to(torch.int32)
    
    # 4. 打包量化权重
    if w_bits == 4:
        packing_factor = 8
        
        # 重新组织：[out_features, num_groups, group_size] -> [padded_in_features, out_features]
        quantized_reshaped = quantized_gptq.view(out_features, -1).t()
        
        # 填充到packing_factor的倍数
        total_elements = quantized_reshaped.shape[0]
        packed_rows = (total_elements + packing_factor - 1) // packing_factor
        padded_size = packed_rows * packing_factor
        
        if total_elements < padded_size:
            padding = torch.zeros(padded_size - total_elements, out_features, 
                                dtype=quantized_reshaped.dtype, device=weight.device)
            quantized_reshaped = torch.cat([quantized_reshaped, padding], dim=0)
        
        # 重塑并打包
        quantized_packed = quantized_reshaped.view(packed_rows, packing_factor, out_features)
        
        qweight = torch.zeros(packed_rows, out_features, dtype=torch.int32, device=weight.device)
        for i in range(packing_factor):
            qweight += (quantized_packed[:, i, :] << (i * 4))
    else:
        raise ValueError(f"不支持的量化位数: {w_bits}")
    
    # 5. 生成qzeros - 使用LSQ的零点偏移
    zero_point = -Qn  # LSQ量化的零点偏移
    qzeros_cols = (out_features + 7) // 8
    
    # 创建零点矩阵并打包
    zeros_matrix = torch.full((num_groups, out_features), zero_point, 
                             dtype=torch.int32, device=weight.device)
    
    # 填充到8的倍数
    if out_features % 8 != 0:
        padding_cols = 8 - (out_features % 8)
        padding = torch.zeros(num_groups, padding_cols, dtype=torch.int32, device=weight.device)
        zeros_matrix = torch.cat([zeros_matrix, padding], dim=1)
    
    # 打包零点
    zeros_packed = zeros_matrix.view(num_groups, qzeros_cols, 8)
    qzeros = torch.zeros(num_groups, qzeros_cols, dtype=torch.int32, device=weight.device)
    for i in range(8):
        qzeros += (zeros_packed[:, :, i] << (i * 4))
    
    # 6. 调整scales格式：直接使用训练时的scale
    # GPTQ scales格式: [num_groups, out_features]
    gptq_scales = scales.t().contiguous()  # 转置：[out_features, num_groups] -> [num_groups, out_features]
    
    # 7. 生成g_idx
    g_idx = torch.zeros(in_features, dtype=torch.int32, device=weight.device)
    for group_id in range(num_groups):
        start_idx = group_id * group_size
        end_idx = min(start_idx + group_size, in_features)
        g_idx[start_idx:end_idx] = group_id
    
    # 验证
    print(f"    简化转换验证:")
    print(f"      原始权重范围: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
    print(f"      LSQ量化范围: [{quantized_lsq.min().item()}, {quantized_lsq.max().item()}]")
    print(f"      GPTQ量化范围: [{quantized_gptq.min().item()}, {quantized_gptq.max().item()}]")
    print(f"      使用训练scales: [{gptq_scales.min().item():.6f}, {gptq_scales.max().item():.6f}]")
    print(f"      零点值: {zero_point}")
    
    return qweight, gptq_scales, qzeros, g_idx


def convert_checkpoint_to_gptq(
    checkpoint_path="/root/autodl-tmp/output/checkpoint-3",
    gptq_template_path="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-format",
    output_path="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-converted",
    use_simple_conversion=True
):
    """
    将分组量化的checkpoint转换为GPTQ格式
    
    Args:
        checkpoint_path: 输入的checkpoint路径
        gptq_template_path: GPTQ模板路径 
        output_path: 输出路径
        use_simple_conversion: 是否使用简化转换方法
    """
    
    print("=== 开始转换checkpoint到GPTQ格式 ===")
    print(f"使用{'简化' if use_simple_conversion else '标准'}转换方法")
    
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
            layer_name = key.replace('.weight_clip_val', '')
            quantized_layers.add(layer_name)
    
    print(f"  找到 {len(quantized_layers)} 个量化层")
    
    # 选择转换函数
    convert_func = convert_weight_to_gptq_simple if use_simple_conversion else convert_weight_to_gptq
    
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
        gptq_weight, gptq_scales, gptq_qzeros, gptq_g_idx = convert_func(
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
    
    # 复制其他非量化参数
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
    parser.add_argument("--simple", action='store_true',
                       help="使用简化转换方法")
    
    args = parser.parse_args()
    
    try:
        convert_checkpoint_to_gptq(
            checkpoint_path=args.checkpoint_path,
            gptq_template_path=args.gptq_template_path, 
            output_path=args.output_path,
            use_simple_conversion=args.simple
        )
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
