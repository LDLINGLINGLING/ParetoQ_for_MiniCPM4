#!/usr/bin/env python3
"""
MiniCPM4-0.5B Weight Analysis Script
This script analyzes the model weights and visualizes the sizes of different matrices.
"""

import json
import numpy as np
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
    Analyze model weights and categorize them by matrix type.
    Returns a dictionary with weight statistics.
    """
    weight_stats = defaultdict(list)
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            shape = tensor.shape
            size = tensor.numel()
            
            # Categorize weights by type
            if 'q_proj' in key:
                weight_stats['q'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)  # bfloat16 = 2 bytes
                })
            elif 'k_proj' in key:
                weight_stats['k'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)
                })
            elif 'v_proj' in key:
                weight_stats['v'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)
                })
            elif 'o_proj' in key:
                weight_stats['o'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)
                })
            elif 'up_proj' in key:
                weight_stats['mlp_up'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)
                })
            elif 'down_proj' in key:
                weight_stats['mlp_down'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)
                })
            elif 'gate_proj' in key:
                weight_stats['mlp_gate'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)
                })
            else:
                weight_stats['other'].append({
                    'name': key,
                    'shape': shape,
                    'size': size,
                    'memory_mb': size * 2 / (1024 * 1024)
                })
    
    return weight_stats

def create_summary_stats(weight_stats):
    """Create summary statistics for each weight type"""
    summary = {}
    
    for weight_type, weights in weight_stats.items():
        if weights:
            total_size = sum(w['size'] for w in weights)
            total_memory = sum(w['memory_mb'] for w in weights)
            avg_size = total_size / len(weights)
            avg_memory = total_memory / len(weights)
            
            summary[weight_type] = {
                'count': len(weights),
                'total_size': total_size,
                'total_memory_mb': total_memory,
                'avg_size': avg_size,
                'avg_memory_mb': avg_memory
            }
    
    return summary

def visualize_weight_analysis(weight_stats, summary_stats, save_dir='.'):
    """Create visualizations for weight analysis"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MiniCPM4-0.5B Weight Analysis', fontsize=16, fontweight='bold')
    
    # 1. Total size by weight type (Bar chart)
    ax1 = axes[0, 0]
    types = list(summary_stats.keys())
    sizes = [summary_stats[t]['total_size'] for t in types]
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    
    bars = ax1.bar(types, sizes, color=colors)
    ax1.set_title('Total Parameters by Weight Type')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_xlabel('Weight Type')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{size:,}', ha='center', va='bottom', fontsize=8)
    
    # 2. Memory usage by weight type (Pie chart)
    ax2 = axes[0, 1]
    memory_sizes = [summary_stats[t]['total_memory_mb'] for t in types]
    wedges, texts, autotexts = ax2.pie(memory_sizes, labels=types, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Memory Usage Distribution (MB)')
    
    # 3. Average parameter count per layer
    ax3 = axes[0, 2]
    avg_sizes = [summary_stats[t]['avg_size'] for t in types]
    ax3.bar(types, avg_sizes, color=colors)
    ax3.set_title('Average Parameters per Layer')
    ax3.set_ylabel('Average Parameters')
    ax3.set_xlabel('Weight Type')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Layer count by weight type
    ax4 = axes[1, 0]
    layer_counts = [summary_stats[t]['count'] for t in types]
    ax4.bar(types, layer_counts, color=colors)
    ax4.set_title('Number of Layers by Weight Type')
    ax4.set_ylabel('Layer Count')
    ax4.set_xlabel('Weight Type')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Detailed heatmap of shapes for main weight types
    ax5 = axes[1, 1]
    main_types = ['q', 'k', 'v', 'o', 'mlp_up', 'mlp_down']
    shape_data = []
    
    for wtype in main_types:
        if wtype in weight_stats and weight_stats[wtype]:
            for weight in weight_stats[wtype]:
                if len(weight['shape']) == 2:  # Only 2D matrices
                    shape_data.append([wtype, weight['shape'][0], weight['shape'][1]])
    
    if shape_data:
        df_shapes = pd.DataFrame(shape_data, columns=['Type', 'Dim1', 'Dim2'])
        pivot_table = df_shapes.pivot_table(values='Dim2', index='Type', columns='Dim1', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='Blues', ax=ax5)
        ax5.set_title('Average Matrix Dimensions Heatmap')
    
    # 6. Memory efficiency comparison
    ax6 = axes[1, 2]
    efficiency = [summary_stats[t]['total_memory_mb'] / summary_stats[t]['count'] 
                 for t in types]
    ax6.bar(types, efficiency, color=colors)
    ax6.set_title('Memory per Layer (MB)')
    ax6.set_ylabel('Memory (MB)')
    ax6.set_xlabel('Weight Type')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/weight_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_statistics(weight_stats, summary_stats, config):
    """Print detailed statistics to console"""
    print("="*80)
    print("MiniCPM4-0.5B Weight Analysis Report")
    print("="*80)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Intermediate Size: {config['intermediate_size']}")
    print(f"  Number of Layers: {config['num_hidden_layers']}")
    print(f"  Number of Attention Heads: {config['num_attention_heads']}")
    print(f"  Number of KV Heads: {config['num_key_value_heads']}")
    print(f"  Vocabulary Size: {config['vocab_size']}")
    
    print(f"\nWeight Statistics Summary:")
    print("-" * 80)
    print(f"{'Weight Type':<15} {'Count':<8} {'Total Params':<15} {'Memory (MB)':<12} {'Avg/Layer':<12}")
    print("-" * 80)
    
    total_params = 0
    total_memory = 0
    
    for weight_type, stats in summary_stats.items():
        total_params += stats['total_size']
        total_memory += stats['total_memory_mb']
        
        print(f"{weight_type:<15} {stats['count']:<8} {stats['total_size']:<15,} "
              f"{stats['total_memory_mb']:<12.2f} {stats['avg_size']:<12,.0f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {'':<8} {total_params:<15,} {total_memory:<12.2f}")
    print(f"\nTotal Model Size: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)")
    print(f"Total Parameters: {total_params:,}")

def main():
    """Main function to run the analysis"""
    model_dir = "D:\model_best\minicpm\pretrain_model\MiniCPM4-0_5B"
    config_path = f"{model_dir}/config.json"
    safetensors_path = f"{model_dir}/model.safetensors"
    
    print("Loading model configuration...")
    config = load_model_config(config_path)
    
    print("Analyzing model weights...")
    weight_stats = analyze_model_weights(safetensors_path)
    
    print("Computing summary statistics...")
    summary_stats = create_summary_stats(weight_stats)
    
    print("Creating visualizations...")
    visualize_weight_analysis(weight_stats, summary_stats, model_dir)
    
    print("Generating detailed report...")
    print_detailed_statistics(weight_stats, summary_stats, config)
    
    print(f"\nAnalysis complete! Visualization saved to {model_dir}/weight_analysis.png")

if __name__ == "__main__":
    main()