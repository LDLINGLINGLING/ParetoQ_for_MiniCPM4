#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPTQ 量化参数初始化使用示例

这个脚本展示了如何使用 init_params.py 中的函数来初始化量化参数：
1. 使用线性量化方法（快速但精度较低）
2. 使用 GPTQ 算法（精度高但需要校准数据）
3. NaN 检测和修复
4. 设置可训练参数

作者：AI Assistant
日期：2025-07-08
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from init_params import (
    initialize_quantization_params,
    initialize_quantization_params_gptq_style,
    initialize_quantization_params_gptq,
    create_calibration_data_from_model_weights,
    create_calibration_data_from_real_data,
    collect_layer_activations,
    check_nan_in_model,
    fix_nan_in_model,
    only_train_adapter
)

def create_sample_quantized_model():
    """创建一个带有量化参数的示例模型"""
    
    class QuantizedLinear(nn.Module):
        def __init__(self, in_features, out_features, group_size=128, bits=4):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.group_size = group_size
            self.bits = bits
            
            # 计算分组数量
            num_groups = (in_features + group_size - 1) // group_size
            
            # 量化参数
            self.scale = nn.Parameter(torch.ones(out_features, num_groups))
            self.zero = nn.Parameter(torch.zeros(out_features, num_groups))
        
        def forward(self, x):
            # 这里只是一个示例，实际的量化前向传播会更复杂
            return self.linear(x)
    
    class SampleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=512, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            
            # 创建多个量化层
            self.layers = nn.ModuleList([
                QuantizedLinear(hidden_size, hidden_size, group_size=128, bits=4)
                for _ in range(num_layers)
            ])
            
            self.output = QuantizedLinear(hidden_size, vocab_size, group_size=128, bits=4)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)
            
            return self.output(x)
    
    return SampleModel()

def example_linear_quantization():
    """示例 1：使用线性量化方法初始化"""
    print("="*60)
    print("示例 1：线性量化参数初始化")
    print("="*60)
    
    # 创建模型
    model = create_sample_quantized_model()
    
    # 检查初始化前的参数
    print("初始化前的参数统计：")
    for name, param in model.named_parameters():
        if 'scale' in name or 'zero' in name:
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    
    # 使用线性量化初始化
    model = initialize_quantization_params(model, group_size=128)
    
    # 检查初始化后的参数
    print("\n初始化后的参数统计：")
    for name, param in model.named_parameters():
        if 'scale' in name or 'zero' in name:
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    
    return model

def example_gptq_style_quantization():
    """示例 2：使用 GPTQ-style 线性量化方法"""
    print("\\n" + "="*60)
    print("示例 2：GPTQ-style 线性量化参数初始化")
    print("="*60)
    
    # 创建模型
    model = create_sample_quantized_model()
    
    # 使用 GPTQ-style 线性量化初始化
    model = initialize_quantization_params_gptq_style(
        model, 
        group_size=128, 
        bits=4, 
        symmetric=False, 
        verbose=True
    )
    
    # 检查初始化后的参数
    print("\\n初始化后的参数统计：")
    for name, param in model.named_parameters():
        if 'scale' in name or 'zero' in name:
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    
    return model

def example_gptq_quantization():
    """示例 3：使用真实的 GPTQ 算法"""
    print("\\n" + "="*60)
    print("示例 3：真实 GPTQ 量化参数初始化")
    print("="*60)
    
    # 创建模型
    model = create_sample_quantized_model()
    
    # 创建校准数据
    print("创建校准数据...")
    calibration_data = create_calibration_data_from_model_weights(
        model, num_samples=32, batch_size=8
    )
    
    try:
        # 使用 GPTQ 量化初始化
        model = initialize_quantization_params_gptq(
            model, 
            calibration_data=calibration_data,
            group_size=128, 
            bits=4, 
            symmetric=False,
            blocksize=128,
            percdamp=0.01,
            actorder=False,
            static_groups=False,
            verbose=True
        )
        
        # 检查初始化后的参数
        print("\\n初始化后的参数统计：")
        for name, param in model.named_parameters():
            if 'scale' in name or 'zero' in name:
                print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
                
    except Exception as e:
        print(f"GPTQ 初始化失败: {e}")
        print("这通常是因为 AutoGPTQ 库未正确安装或校准数据不合适")
        
        # 回退到线性量化
        print("\\n回退到线性量化...")
        model = initialize_quantization_params_gptq_style(
            model, group_size=128, bits=4, symmetric=False, verbose=True
        )
    
    return model

def example_nan_detection_and_repair():
    """示例 4：NaN 检测和修复"""
    print("\\n" + "="*60)
    print("示例 4：NaN 检测和修复")
    print("="*60)
    
    # 创建模型
    model = create_sample_quantized_model()
    
    # 人工引入 NaN 值
    print("人工引入 NaN 值...")
    for name, param in model.named_parameters():
        if 'scale' in name:
            param.data[0, 0] = float('nan')
            break
    
    # 检测 NaN
    print("\\n检测 NaN 值...")
    has_nan, nan_info = check_nan_in_model(model, detailed=True)
    
    if has_nan:
        print("\\n修复 NaN 值...")
        model, fixed_params, total_nan_count = fix_nan_in_model(model, verbose=True)
        print(f"修复了 {total_nan_count} 个 NaN 值，涉及 {fixed_params} 个参数")
    
    return model

def example_set_trainable_params():
    """示例 5：设置可训练参数"""
    print("\\n" + "="*60)
    print("示例 5：设置仅量化参数可训练")
    print("="*60)
    
    # 创建模型
    model = create_sample_quantized_model()
    
    # 初始化量化参数
    model = initialize_quantization_params(model, group_size=128)
    
    # 设置仅量化参数可训练
    model = only_train_adapter(model, verbose=True)
    
    # 检查可训练参数
    print("\\n可训练参数统计：")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  可训练参数数量: {trainable_params:,}")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数比例: {trainable_params/total_params*100:.2f}%")
    
    return model

def example_with_huggingface_model():
    """示例 6：使用 Hugging Face 模型（需要相应的模型和tokenizer）"""
    print("\\n" + "="*60)
    print("示例 6：使用 Hugging Face 模型")
    print("="*60)
    
    try:
        # 加载预训练模型和tokenizer
        # 注意：这里需要一个实际存在的模型，您可能需要调整路径
        model_name = "microsoft/DialoGPT-small"  # 使用一个较小的模型作为示例
        
        print(f"尝试加载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 准备校准文本
        calibration_texts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Can you help me with this problem?",
            "This is a test sentence for calibration."
        ]
        
        # 创建校准数据
        calibration_data = create_calibration_data_from_real_data(
            model, tokenizer, calibration_texts, max_length=128, batch_size=2
        )
        
        print(f"创建了 {len(calibration_data)} 个校准数据批次")
        
        # 注意：这里的模型可能没有量化参数，这只是一个示例
        # 在实际使用中，您需要先添加量化参数到模型中
        
    except Exception as e:
        print(f"加载 Hugging Face 模型失败: {e}")
        print("这通常是因为模型不存在或网络问题")
        print("在实际使用中，请确保模型路径正确且已安装 transformers 库")

def main():
    """主函数：运行所有示例"""
    print("GPTQ 量化参数初始化使用示例")
    print("="*60)
    
    # 运行所有示例
    model1 = example_linear_quantization()
    model2 = example_gptq_style_quantization()
    model3 = example_gptq_quantization()
    model4 = example_nan_detection_and_repair()
    model5 = example_set_trainable_params()
    
    # 可选：运行 Hugging Face 模型示例
    example_with_huggingface_model()
    
    print("\\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    
    print("""
使用建议：
1. 对于快速原型和测试，使用线性量化方法
2. 对于生产环境，推荐使用 GPTQ 算法（如果有合适的校准数据）
3. 始终检查和修复 NaN 值
4. 训练时只让量化参数可训练，以提高效率
5. 使用真实的校准数据而不是随机数据
""")

if __name__ == "__main__":
    main()
