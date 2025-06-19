# 分组量化使用指南

## 概述

分组量化是一种先进的权重量化技术，它将权重矩阵按组进行量化，每组使用独立的缩放因子。相比传统的逐行量化，分组量化能够：

- 提高量化精度，减少量化误差
- 更好地保持权重分布的细节信息
- 在相同位数下获得更好的模型性能

## 配置参数

### 基本参数

- `enable_groupwise`: 是否启用分组量化 (True/False)
- `group_size`: 分组大小，通常选择 32、64、128 等值
- `w_bits`: 量化位数 (1-4 位)

### 推荐配置

#### 4位分组量化 (推荐)
```bash
--w_bits 4
--enable_groupwise True
--group_size 128
```

#### 3位分组量化
```bash
--w_bits 3
--enable_groupwise True  
--group_size 64
```

#### 2位分组量化
```bash
--w_bits 2
--enable_groupwise True
--group_size 32
```

## 使用示例

### 1. 命令行使用

```bash
cd /root/autodl-tmp/ParetoQ_for_MiniCPM4

python train_minicpm.py \
    --input_model_filename "/root/autodl-tmp/MiniCPM4-0_5B" \
    --output_model_filename "minicpm_4bit_grouped" \
    --w_bits 4 \
    --enable_groupwise True \
    --group_size 128 \
    --train_data_local_path "training_dataset_example.jsonl" \
    --num_train_epochs 3 \
    --learning_rate 5e-5
```

### 2. 脚本配置

修改 `run_train_minicpm.sh` 中的相关参数：

```bash
# 启用4位分组量化
--w_bits 4 \
--enable_groupwise True \
--group_size 128 \
```

### 3. 调试模式配置

在 `train_minicpm.py` 的 `setup_debug_args()` 函数中：

```python
model_args.w_bits = 4
model_args.group_size = 128
model_args.enable_groupwise = True
```

## 性能对比

### 内存使用
- 传统量化: 每行一个缩放因子
- 分组量化: 每组一个缩放因子 (内存增加 input_dim/group_size 倍)

### 量化精度
- 分组量化通常比传统量化精度更高
- 更小的组大小 = 更高精度，但更多内存开销

### 推荐组大小选择

| 量化位数 | 推荐组大小 | 内存开销 | 精度 |
|---------|-----------|---------|------|
| 4-bit   | 128       | 低      | 高   |
| 3-bit   | 64        | 中      | 高   |
| 2-bit   | 32        | 高      | 中   |
| 1-bit   | 32        | 高      | 低   |

## 技术原理

### 分组过程

1. 将权重矩阵 W[out_features, in_features] 重塑为 W[out_features, num_groups, group_size]
2. 为每组计算独立的缩放因子 α[out_features, num_groups]
3. 每组内的权重使用相同的缩放因子进行量化

### 量化公式

对于每个组 g：
```
W_quantized[g] = Quantize(W[g] / α[g]) * α[g]
```

其中 α[g] 是第g组的缩放因子。

### 梯度计算

分组量化的梯度计算考虑了每组独立的缩放因子，确保训练过程中的梯度传播正确。

## 注意事项

1. **内存开销**: 分组量化会增加缩放因子的内存使用
2. **组大小选择**: 应根据模型大小和可用内存选择合适的组大小
3. **量化位数**: 建议与分组量化配合使用3-4位量化获得最佳效果
4. **训练稳定性**: 初始训练阶段可能需要较小的学习率

## 故障排除

### 常见问题

1. **显存不足**
   - 减小 group_size
   - 减少 batch_size
   - 使用更少的量化位数

2. **训练不稳定**
   - 降低学习率
   - 增加预热步数
   - 检查梯度裁剪设置

3. **精度下降**
   - 减小 group_size
   - 增加量化位数
   - 调整训练轮数

### 性能优化建议

1. 对于大模型，使用 group_size=128 或更大
2. 对于小模型，可以使用 group_size=32 或 64
3. 配合 bf16 混合精度训练
4. 使用梯度累积减少内存压力