# GPTQ 量化参数初始化工具

本工具提供了多种量化参数初始化方法，包括线性量化和基于 GPTQ 算法的高精度量化。

## 功能特性

- **线性量化初始化**：快速的基于统计的量化参数初始化
- **GPTQ 量化初始化**：使用 AutoGPTQ 库的高精度量化算法
- **自动 Fallback**：当 GPTQ 不可用时自动回退到线性量化
- **NaN 检测和修复**：自动检测和修复模型中的 NaN 值
- **校准数据生成**：从模型权重或真实数据生成校准数据
- **参数管理**：设置仅量化参数可训练

## 文件结构

```
utils/
├── init_params.py          # 主要的量化参数初始化函数
├── example_usage.py        # 使用示例
└── README.md              # 本文档
```

## 主要函数

### 1. 基本量化参数初始化

```python
def initialize_quantization_params(model, group_size=None):
    """
    初始化模型中的量化参数（scale和zero），支持分组量化
    
    Args:
        model: PyTorch模型
        group_size: 分组大小，如果为None则使用全局量化
        
    Returns:
        model: 初始化后的模型
    """
```

### 2. GPTQ-style 线性量化

```python
def initialize_quantization_params_gptq_style(
    model, 
    group_size=None, 
    bits=8, 
    symmetric=False, 
    verbose=True
):
    """
    使用线性量化方法初始化量化参数（GPTQ-style，但不使用Hessian）
    
    Args:
        model: 要量化的PyTorch模型
        group_size: 分组大小，如果为None则使用全局量化
        bits: 量化位数
        symmetric: 是否使用对称量化
        verbose: 是否打印详细信息
        
    Returns:
        model: 量化参数初始化后的模型
    """
```

### 3. 真实 GPTQ 量化

```python
def initialize_quantization_params_gptq(
    model, 
    calibration_data=None, 
    group_size=None, 
    bits=8, 
    symmetric=False, 
    blocksize=128, 
    percdamp=0.01, 
    actorder=False, 
    static_groups=False,
    verbose=True
):
    """
    使用GPTQ算法初始化量化参数
    
    Args:
        model: 要量化的PyTorch模型
        calibration_data: 校准数据
        group_size: 分组大小
        bits: 量化位数
        symmetric: 是否使用对称量化
        blocksize: GPTQ算法的块大小
        percdamp: 阻尼系数百分比
        actorder: 是否使用激活顺序
        static_groups: 是否使用静态分组
        verbose: 是否打印详细信息
        
    Returns:
        model: 量化参数初始化后的模型
    """
```

### 4. 校准数据生成

```python
def create_calibration_data_from_model_weights(model, num_samples=32, batch_size=8):
    """从模型权重创建校准数据的简化版本"""

def create_calibration_data_from_real_data(model, tokenizer, texts, max_length=512, batch_size=8):
    """从真实文本数据创建校准数据"""
```

### 5. NaN 检测和修复

```python
def check_nan_in_model(model, detailed=False):
    """检查模型中的NaN值"""

def fix_nan_in_model(model, verbose=True, inplace=True):
    """自动修复模型中包含NaN值的参数"""
```

### 6. 参数管理

```python
def only_train_adapter(model, verbose=True):
    """设置仅量化参数（scale和zero）可训练"""
```

## 使用示例

### 快速开始

```python
import torch
import torch.nn as nn
from init_params import initialize_quantization_params

# 创建带有量化参数的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)
        self.linear.scale = nn.Parameter(torch.ones(64, 4))  # 假设分组大小为32
        self.linear.zero = nn.Parameter(torch.zeros(64, 4))
    
    def forward(self, x):
        return self.linear(x)

# 初始化量化参数
model = MyModel()
model = initialize_quantization_params(model, group_size=32)
```

### 使用 GPTQ 算法

```python
from init_params import (
    initialize_quantization_params_gptq,
    create_calibration_data_from_model_weights
)

# 创建校准数据
calibration_data = create_calibration_data_from_model_weights(
    model, num_samples=32, batch_size=8
)

# 使用 GPTQ 算法初始化
model = initialize_quantization_params_gptq(
    model, 
    calibration_data=calibration_data,
    group_size=128, 
    bits=4, 
    symmetric=False,
    verbose=True
)
```

### 完整的工作流程

```python
from init_params import *

# 1. 创建模型
model = create_your_model()

# 2. 初始化量化参数
model = initialize_quantization_params_gptq(
    model, 
    calibration_data=calibration_data,
    group_size=128, 
    bits=4, 
    symmetric=False,
    verbose=True
)

# 3. 检查和修复 NaN 值
has_nan, nan_info = check_nan_in_model(model, detailed=True)
if has_nan:
    model, fixed_params, total_nan_count = fix_nan_in_model(model, verbose=True)

# 4. 设置仅量化参数可训练
model = only_train_adapter(model, verbose=True)

# 5. 开始训练
# ... 训练代码 ...
```

## 依赖项

- PyTorch
- AutoGPTQ（可选，用于真实 GPTQ 算法）
- transformers（可选，用于 Hugging Face 模型）

## 安装 AutoGPTQ

```bash
pip install auto-gptq
```

或者从源码安装：

```bash
git clone https://github.com/PanQiWei/AutoGPTQ.git
cd AutoGPTQ
pip install -e .
```

## 注意事项

1. **校准数据**：使用真实的输入数据作为校准数据，而不是随机数据，以获得更好的量化效果。

2. **模型结构**：确保您的模型包含相应的量化参数（scale 和 zero）。

3. **内存使用**：GPTQ 算法需要额外的内存来存储 Hessian 矩阵，对于大型模型可能需要较多内存。

4. **自动 Fallback**：如果 AutoGPTQ 不可用或没有校准数据，系统会自动回退到线性量化方法。

5. **参数命名**：量化参数应该以 `.scale` 和 `.zero` 结尾，例如 `layer1.scale`、`layer1.zero`。

6. **NaN 处理**：训练过程中可能出现 NaN 值，建议定期检查和修复。

## 故障排除

### 常见问题

1. **AutoGPTQ 导入失败**
   - 确保已正确安装 AutoGPTQ
   - 检查 CUDA 版本兼容性

2. **形状不匹配**
   - 检查量化参数的形状是否与分组大小匹配
   - 确保 `group_size` 参数设置正确

3. **NaN 值**
   - 使用 `check_nan_in_model` 检查
   - 使用 `fix_nan_in_model` 修复

4. **内存不足**
   - 减少校准数据的批大小
   - 使用更小的 blocksize

## 许可证

本工具遵循 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

- **2025-07-08**: 初始版本发布
  - 支持线性量化和 GPTQ 算法
  - 自动 Fallback 机制
  - NaN 检测和修复
  - 校准数据生成
  - 参数管理功能
