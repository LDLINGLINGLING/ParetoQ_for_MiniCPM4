#!/usr/bin/env python3
"""
测试转换后的GPTQ模型困惑度
"""

import torch
import json
import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import numpy as np

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def load_test_data(data_path, max_samples=None):
    """加载测试数据"""
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line.strip())
            texts.append(data['text'])
    return texts

def calculate_perplexity(model, dataloader, device):
    """计算困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算困惑度"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 创建labels，忽略padding token
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # 忽略padding部分的损失
            
            try:
                # 计算损失
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                loss = outputs.loss
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 检测到无效损失值: {loss.item()}")
                    continue
                
                # 计算有效token数量（排除padding和忽略的token）
                num_valid_tokens = (labels != -100).sum().item()
                if num_valid_tokens > 0:
                    total_loss += loss.item() * num_valid_tokens
                    total_tokens += num_valid_tokens
                    
            except Exception as e:
                print(f"批次处理出错: {e}")
                continue
    
    if total_tokens == 0:
        print("错误: 没有有效的token用于计算困惑度")
        return float('inf'), float('inf')
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    
    # 检查平均损失的有效性
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        print(f"错误: 平均损失无效: {avg_loss}")
        return float('inf'), avg_loss
    
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')
    
    return perplexity, avg_loss

def test_model_perplexity(
    model_path="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-converted",
    data_path="/root/autodl-tmp/ParetoQ_for_MiniCPM4/training_dataset_example.jsonl",
    max_samples=100,
    batch_size=4,
    max_length=512,
    device=None
):
    """测试模型困惑度"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== GPTQ模型困惑度测试 ===")
    print(f"模型路径: {model_path}")
    print(f"数据路径: {data_path}")
    print(f"设备: {device}")
    print(f"最大样本数: {max_samples}")
    print(f"批次大小: {batch_size}")
    print(f"最大长度: {max_length}")
    print()
    
    # 1. 加载tokenizer
    print("1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   词汇表大小: {len(tokenizer)}")
    
    # 2. 加载模型
    print("2. 加载GPTQ模型...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    load_time = time.time() - start_time
    print(f"   模型加载完成，耗时: {load_time:.2f}秒")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 3. 加载测试数据
    print("3. 加载测试数据...")
    texts = load_test_data(data_path, max_samples)
    print(f"   加载了 {len(texts)} 个样本")
    
    # 4. 创建数据集和数据加载器
    print("4. 创建数据集...")
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"   创建了 {len(dataloader)} 个批次")
    
    # 5. 计算困惑度
    print("5. 计算困惑度...")
    start_time = time.time()
    perplexity, avg_loss = calculate_perplexity(model, dataloader, device)
    calc_time = time.time() - start_time
    
    # 6. 打印结果
    print("\n=== 测试结果 ===")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"困惑度: {perplexity:.4f}")
    print(f"计算时间: {calc_time:.2f}秒")
    print(f"平均每样本时间: {calc_time/len(texts):.3f}秒")
    
    # 7. 保存结果
    results = {
        "model_path": model_path,
        "data_path": data_path,
        "num_samples": len(texts),
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "calc_time": calc_time,
        "batch_size": batch_size,
        "max_length": max_length
    }
    
    results_path = "/root/autodl-tmp/gptq_perplexity_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_path}")
    
    return perplexity, avg_loss

def compare_with_baseline():
    """与基线模型比较"""
    print("\n=== 与训练结果比较 ===")
    
    # 读取训练时的结果
    try:
        with open("/root/autodl-tmp/output/eval_results.json", 'r') as f:
            train_results = json.load(f)
        train_perplexity = train_results.get("perplexity", None)
        
        if train_perplexity:
            print(f"训练时困惑度: {train_perplexity:.4f}")
            print("注意: 训练时使用的是验证集，当前测试使用的是示例数据")
        else:
            print("未找到训练时的困惑度数据")
    except FileNotFoundError:
        print("未找到训练结果文件")

def main():
    parser = argparse.ArgumentParser(description="测试GPTQ模型困惑度")
    parser.add_argument("--model_path", 
                       default="/root/autodl-tmp/MiniCPM4-0.5B-QAT-Int4-GPTQ-converted",
                       help="GPTQ模型路径")
    parser.add_argument("--data_path",
                       default="/root/autodl-tmp/ParetoQ_for_MiniCPM4/training_dataset_example.jsonl",
                       help="测试数据路径")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="最大测试样本数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--device", default=None,
                       help="计算设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        # 测试困惑度
        perplexity, avg_loss = test_model_perplexity(
            model_path=args.model_path,
            data_path=args.data_path,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device
        )
        
        # 与基线比较
        compare_with_baseline()
        
        print(f"\n=== 最终结果 ===")
        print(f"GPTQ模型困惑度: {perplexity:.4f}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
