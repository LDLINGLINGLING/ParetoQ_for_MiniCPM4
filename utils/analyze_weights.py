#!/usr/bin/env python3
"""
MiniCPM4-0.5B Weight Analysis Script
This script analyzes the model weight values and visualizes their distributions.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端以确保图片显示
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors import safe_open
from collections import defaultdict
import pandas as pd

def load_model_config(config_path):
    """Load model configuration from config.json"""
    with open(config_path, 'r') as f:
        return json.load(f)

def analyze_model_weights(safetensors_path):
    """
    Analyze model weight values and categorize them by matrix type.
    Returns a dictionary with weight value statistics.
    """
    weight_stats = defaultdict(list)
    
    # 仅分析指定的矩阵类型
    target_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # 只处理目标矩阵类型
            matrix_type = None
            for target in target_types:
                if target in key:
                    matrix_type = target.replace('_proj', '')
                    if matrix_type == 'up':
                        matrix_type = 'mlp_up'
                    elif matrix_type == 'down':
                        matrix_type = 'mlp_down'
                    break
            
            if matrix_type:
                # 修复BFloat16转换问题：先转为float32再转numpy
                weight_values = tensor.float().cpu().numpy().flatten()
                
                weight_stats[matrix_type].append({
                    'name': key,
                    'shape': tensor.shape,
                    'values': weight_values,
                    'layer_idx': extract_layer_index(key)
                })
    
    return weight_stats

def extract_layer_index(weight_name):
    """Extract layer index from weight name"""
    import re
    match = re.search(r'layers\.(\d+)\.', weight_name)
    return int(match.group(1)) if match else -1

def create_summary_stats(weight_stats):
    """Create summary statistics for weight values"""
    summary = {}
    
    for weight_type, weights in weight_stats.items():
        if weights:
            # 合并所有该类型的权重值
            all_values = np.concatenate([w['values'] for w in weights])
            
            # 计算各层的统计量
            layer_means = [np.mean(w['values']) for w in weights]
            layer_stds = [np.std(w['values']) for w in weights]
            layer_mins = [np.min(w['values']) for w in weights]
            layer_maxs = [np.max(w['values']) for w in weights]
            layer_abs_means = [np.mean(np.abs(w['values'])) for w in weights]
            
            summary[weight_type] = {
                'count': len(weights),
                'all_values': all_values,
                'global_mean': np.mean(all_values),
                'global_std': np.std(all_values),
                'global_min': np.min(all_values),
                'global_max': np.max(all_values),
                'global_abs_mean': np.mean(np.abs(all_values)),
                'percentile_1': np.percentile(all_values, 1),
                'percentile_5': np.percentile(all_values, 5),
                'percentile_25': np.percentile(all_values, 25),
                'percentile_50': np.percentile(all_values, 50),  # median
                'percentile_75': np.percentile(all_values, 75),
                'percentile_95': np.percentile(all_values, 95),
                'percentile_99': np.percentile(all_values, 99),
                'layer_means': layer_means,
                'layer_stds': layer_stds,
                'layer_mins': layer_mins,
                'layer_maxs': layer_maxs,
                'layer_abs_means': layer_abs_means,
                'zero_ratio': np.sum(all_values == 0) / len(all_values)
            }
    
    return summary

def visualize_weight_analysis(weight_stats, summary_stats, save_dir='.'):
    """Create visualizations for weight value analysis with box plots and distributions"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('MiniCPM4-0.5B Weight Value Distribution Analysis', fontsize=16, fontweight='bold')
    
    types = list(summary_stats.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    
    # 1. 权重值箱型图
    ax1 = axes[0, 0]
    
    # 为了避免极端值影响可视化，我们使用百分位数进行截断
    box_data = []
    for t in types:
        values = summary_stats[t]['all_values']
        # 使用1%-99%百分位数进行截断
        p1, p99 = np.percentile(values, [1, 99])
        filtered_values = values[(values >= p1) & (values <= p99)]
        box_data.append(filtered_values[:10000] if len(filtered_values) > 10000 else filtered_values)  # 采样以提高性能
    
    box_plot1 = ax1.boxplot(box_data, tick_labels=types, patch_artist=True)
    for patch, color in zip(box_plot1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_title('Weight Value Distribution (1%-99% percentile)')
    ax1.set_ylabel('Weight Values')
    ax1.set_xlabel('Matrix Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. 权重值直方图
    ax2 = axes[0, 1]
    for i, (t, color) in enumerate(zip(types, colors)):
        values = summary_stats[t]['all_values']
        # 使用1%-99%百分位数进行截断
        p1, p99 = np.percentile(values, [1, 99])
        filtered_values = values[(values >= p1) & (values <= p99)]
        ax2.hist(filtered_values, bins=50, alpha=0.6, label=t, color=color, density=True)
    
    ax2.set_title('Weight Value Histogram (Normalized)')
    ax2.set_xlabel('Weight Values')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 各层权重均值的箱型图
    ax3 = axes[0, 2]
    means_data = [summary_stats[t]['layer_means'] for t in types]
    
    box_plot3 = ax3.boxplot(means_data, tick_labels=types, patch_artist=True)
    for patch, color in zip(box_plot3['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_title('Layer-wise Mean Values Distribution')
    ax3.set_ylabel('Mean Weight Values')
    ax3.set_xlabel('Matrix Type')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 各层权重标准差的箱型图
    ax4 = axes[1, 0]
    stds_data = [summary_stats[t]['layer_stds'] for t in types]
    
    box_plot4 = ax4.boxplot(stds_data, tick_labels=types, patch_artist=True)
    for patch, color in zip(box_plot4['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_title('Layer-wise Standard Deviation Distribution')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_xlabel('Matrix Type')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. 权重绝对值均值比较
    ax5 = axes[1, 1]
    abs_means = [summary_stats[t]['global_abs_mean'] for t in types]
    bars = ax5.bar(types, abs_means, color=colors)
    ax5.set_title('Average Absolute Weight Values')
    ax5.set_ylabel('Mean |Weight|')
    ax5.set_xlabel('Matrix Type')
    ax5.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, val in zip(bars, abs_means):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 6. 权重范围（min-max）比较
    ax6 = axes[1, 2]
    min_vals = [summary_stats[t]['global_min'] for t in types]
    max_vals = [summary_stats[t]['global_max'] for t in types]
    
    x_pos = np.arange(len(types))
    width = 0.35
    
    ax6.bar(x_pos - width/2, min_vals, width, label='Min', color='lightcoral', alpha=0.7)
    ax6.bar(x_pos + width/2, max_vals, width, label='Max', color='lightblue', alpha=0.7)
    
    ax6.set_title('Weight Value Ranges')
    ax6.set_ylabel('Weight Values')
    ax6.set_xlabel('Matrix Type')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(types, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 确保图片保存和显示
    save_path = f'{save_dir}/weight_distribution_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # 强制显示图片
    plt.show(block=True)

def print_detailed_statistics(weight_stats, summary_stats, config):
    """Print detailed statistics to console"""
    print("="*100)
    print("MiniCPM4-0.5B Weight Value Distribution Analysis Report")
    print("="*100)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Intermediate Size: {config['intermediate_size']}")
    print(f"  Number of Layers: {config['num_hidden_layers']}")
    print(f"  Number of Attention Heads: {config['num_attention_heads']}")
    print(f"  Number of KV Heads: {config['num_key_value_heads']}")
    print(f"  Vocabulary Size: {config['vocab_size']}")
    
    print(f"\nWeight Value Statistics:")
    print("-" * 120)
    print(f"{'Type':<10} {'Count':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'|Mean|':<10} {'Zero%':<8} {'P50':<10}")
    print("-" * 120)
    
    for weight_type, stats in summary_stats.items():
        print(f"{weight_type:<10} {stats['count']:<6} {stats['global_mean']:<10.6f} "
              f"{stats['global_std']:<10.6f} {stats['global_min']:<10.6f} {stats['global_max']:<10.6f} "
              f"{stats['global_abs_mean']:<10.6f} {stats['zero_ratio']*100:<8.2f} {stats['percentile_50']:<10.6f}")
    
    print("-" * 120)
    
    print(f"\nPercentile Analysis:")
    print("-" * 100)
    print(f"{'Type':<10} {'P1':<10} {'P5':<10} {'P25':<10} {'P50':<10} {'P75':<10} {'P95':<10} {'P99':<10}")
    print("-" * 100)
    
    for weight_type, stats in summary_stats.items():
        print(f"{weight_type:<10} {stats['percentile_1']:<10.6f} {stats['percentile_5']:<10.6f} "
              f"{stats['percentile_25']:<10.6f} {stats['percentile_50']:<10.6f} {stats['percentile_75']:<10.6f} "
              f"{stats['percentile_95']:<10.6f} {stats['percentile_99']:<10.6f}")
    
    print("-" * 100)
    
    # 打印每种类型的详细信息
    print(f"\nDetailed Layer-wise Analysis:")
    for weight_type, weights in weight_stats.items():
        print(f"\n{weight_type.upper()} Matrices:")
        print(f"  Number of layers: {len(weights)}")
        
        layer_stats = summary_stats[weight_type]
        print(f"  Layer means - Min: {np.min(layer_stats['layer_means']):.6f}, "
              f"Max: {np.max(layer_stats['layer_means']):.6f}, "
              f"Std: {np.std(layer_stats['layer_means']):.6f}")
        print(f"  Layer stds  - Min: {np.min(layer_stats['layer_stds']):.6f}, "
              f"Max: {np.max(layer_stats['layer_stds']):.6f}, "
              f"Avg: {np.mean(layer_stats['layer_stds']):.6f}")
        
        # 显示几个示例层
        for i, weight in enumerate(weights[:3]):
            values = weight['values']
            print(f"  Layer {weight['layer_idx']}: shape={weight['shape']}, "
                  f"mean={np.mean(values):.6f}, std={np.std(values):.6f}")

def main():
    """Main function to run the analysis"""
    model_dir = "D:\model_best\minicpm\pretrain_model\Qwen3-0___6B-Base"
    config_path = f"{model_dir}/config.json"
    safetensors_path = f"{model_dir}/model.safetensors"
    
    print("Loading model configuration...")
    config = load_model_config(config_path)
    
    print("Analyzing weight value distributions (q, k, v, o, mlp_up, mlp_down)...")
    weight_stats = analyze_model_weights(safetensors_path)
    
    print("Computing weight value statistics...")
    summary_stats = create_summary_stats(weight_stats)
    
    print("Creating distribution visualizations...")
    visualize_weight_analysis(weight_stats, summary_stats, model_dir)
    
    print("Generating detailed report...")
    print_detailed_statistics(weight_stats, summary_stats, config)
    
    print(f"\nAnalysis complete! Distribution visualization saved to {model_dir}/weight_distribution_analysis.png")

if __name__ == "__main__":
    main()