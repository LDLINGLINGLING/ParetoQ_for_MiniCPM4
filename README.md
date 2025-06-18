# ParetoQ - 极低比特LLM量化训练框架

## 项目简介

ParetoQ 是一个用于大语言模型(LLM)极低比特量化训练的统一框架。该项目支持 1-bit、1.58-bit、2-bit、3-bit 和 4-bit 等多种量化位数，旨在显著减少模型参数大小和计算复杂度，同时保持较好的模型性能。

本项目特别针对 **MiniCPM** 模型进行了优化，提供了完整的量化训练流程，包括模型加载、权重量化、训练和评估等功能。

## 主要特性

- 🚀 **多位量化支持**: 支持1-4位的灵活量化配置
- 🔧 **MiniCPM专用**: 针对MiniCPM模型架构深度优化
- 💾 **内存高效**: 通过量化大幅减少GPU内存使用
- ⚡ **训练加速**: 支持DeepSpeed分布式训练加速
- 📊 **完整流程**: 从数据预处理到模型评估的端到端解决方案
- 🛠️ **易于使用**: 提供脚本化部署和调试模式

## 系统要求

### 硬件要求
- GPU: 支持CUDA的GPU（推荐8GB以上显存）
- 内存: 16GB以上系统内存
- 存储: 至少50GB可用存储空间

### 软件要求
- Python 3.8+
- CUDA 11.0+
- Linux操作系统（推荐Ubuntu 18.04+）

## 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd /root/autodl-tmp
```

### 2. 安装依赖
```bash
pip install -r ParetoQ-main/requirement.txt
```

核心依赖包括：
- `transformers==4.48.3` - Hugging Face变换器库
- `torch>=2.0.0` - PyTorch深度学习框架
- `accelerate>=0.26.0` - 训练加速库
- `datasets==2.20.0` - 数据集处理
- `sentencepiece` - 分词器
- `tensorboardX` - 训练可视化

### 3. 可选：安装DeepSpeed（推荐）
```bash
pip install deepspeed
```

## 项目结构

```
/root/autodl-tmp/
├── ParetoQ-main/                 # 主项目目录
│   ├── train_minicpm.py         # 主训练脚本
│   ├── run_train_minicpm.sh     # 训练启动脚本
│   ├── models/                  # 模型定义
│   │   ├── configuration_minicpm.py
│   │   ├── modeling_minicpm.py
│   │   └── utils_quant.py
│   ├── utils/                   # 工具函数
│   │   ├── datautils.py
│   │   ├── process_args.py
│   │   └── utils.py
│   └── training_dataset_example.jsonl
├── MiniCPM4-0_5B/              # 原始模型
├── Minicpm_quant/              # 量化后模型输出
├── output/                     # 训练输出
├── cache/                      # 缓存目录
└── logs/                       # 训练日志
```

## 使用方法

### 快速开始

1. **准备模型和数据**
   确保以下目录存在：
   - `/root/autodl-tmp/MiniCPM4-0_5B/` - 包含原始MiniCPM模型
   - `/root/autodl-tmp/ParetoQ-main/training_dataset_example.jsonl` - 训练数据

2. **运行训练**
   ```bash
   cd /root/autodl-tmp
   bash run_train_minicpm.sh
   ```

### 详细配置

#### 训练参数配置

主要训练参数可以在 `run_train_minicpm.sh` 中修改：

```bash
# 量化配置
--w_bits 4                      # 量化位数 (1,2,3,4)
--contain_weight_clip_val False  # 是否包含权重裁剪值

# 数据配置
--model_max_length 2048         # 最大序列长度
--per_device_train_batch_size 1 # 每设备训练批次大小
--gradient_accumulation_steps 8 # 梯度累积步数

# 训练配置  
--num_train_epochs 3            # 训练轮数
--learning_rate 5e-5            # 学习率
--warmup_steps 100              # 预热步数
--save_steps 500                # 保存间隔
--eval_steps 500                # 评估间隔
```

#### 调试模式

项目支持调试模式，无需命令行参数即可运行：

```python
# 直接运行Python脚本进行调试
cd /root/autodl-tmp/ParetoQ-main
python train_minicpm.py
```

调试模式会使用预设的默认参数，便于开发和测试。

### 数据格式

训练数据采用JSONL格式，每行一个JSON对象：

```json
{"text": "这是一个训练样本的文本内容"}
{"text": "这是另一个训练样本"}
```

### 量化位数说明

- **1-bit**: 使用权重绝对值均值作为缩放因子
- **2-bit**: 使用权重绝对值最大值作为缩放因子  
- **3-bit/4-bit**: 基于量化范围计算缩放因子

不同位数的量化策略在代码中自动处理，用户只需指定 `w_bits` 参数。

## 输出说明

### 模型输出
- **量化模型**: 保存在 `/root/autodl-tmp/models/minicpm_quantized/`
- **检查点**: 保存在 `/root/autodl-tmp/output/checkpoint-*/`

### 训练日志
- **TensorBoard日志**: `/root/autodl-tmp/logs/`
- **训练指标**: `/root/autodl-tmp/output/all_results.json`
- **评估结果**: `/root/autodl-tmp/output/eval_results.json`

## 性能监控

训练过程中可以通过以下方式监控：

```bash
# 查看GPU使用情况
nvidia-smi

# 查看训练日志
tail -f /root/autodl-tmp/logs/events.out.tfevents.*

# 启动TensorBoard（可选）
tensorboard --logdir=/root/autodl-tmp/logs
```

## 常见问题

### Q: 训练时出现显存不足怎么办？
A: 可以调整以下参数：
- 减少 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps` 
- 减少 `model_max_length`
- 使用更低的量化位数

### Q: 如何使用自己的数据集？
A: 将数据转换为JSONL格式，并修改脚本中的数据路径：
```bash
--train_data_local_path "your_data.jsonl"
--eval_data_local_path "your_eval_data.jsonl"
```

### Q: 支持哪些MiniCPM模型？
A: 理论上支持所有MiniCPM架构的模型，需要确保模型配置文件兼容。

### Q: 如何恢复中断的训练？
A: 训练会自动保存检查点，重新运行脚本会从最新检查点恢复。

## 技术原理

ParetoQ采用以下技术实现高效的量化训练：

1. **权重量化**: 将全精度权重量化到指定位数
2. **梯度缩放**: 自动计算和应用权重缩放因子
3. **内存优化**: 使用低CPU内存模式加载模型
4. **混合精度**: 支持bfloat16训练加速

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。请确保：

1. 代码符合项目风格
2. 添加必要的测试和文档
3. 提交前运行现有测试

## 许可证

本项目基于BSD许可证开源，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本项目专为研究和教育目的设计，请合理使用计算资源，遵守相关法律法规。