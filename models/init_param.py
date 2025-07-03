import torch
import torch.nn as nn
import re

def initialize_quantization_params(model):
    """
    初始化模型中的量化参数（scale和zero）
    
    Args:
        model: PyTorch模型
        
    Returns:
        model: 初始化后的模型
    """
    
    def get_weight_param_name(scale_name):
        """从scale参数名获取对应的weight参数名"""
        # 将.scale替换为.weight，或将.zero替换为.weight
        weight_name = scale_name.replace('.scale', '.weight').replace('.zero', '.weight')
        return weight_name
    
    def symmetric_quantization(weight, bits=8):
        """对称量化：计算scale"""
        max_val = torch.max(torch.abs(weight))
        scale = max_val / (2**(bits-1) - 1)
        return scale
    
    def asymmetric_quantization(weight, bits=8):
        """非对称量化：计算scale和zero"""
        min_val = torch.min(weight)
        max_val = torch.max(weight)
        
        scale = (max_val - min_val) / (2**bits - 1)
        zero = torch.round(-min_val / scale)
        
        return scale, zero
    
    def replace_nan_with_mean(tensor):
        """将tensor中的nan值替换为非nan值的平均值"""
        if torch.isnan(tensor).any():
            nan_mask = torch.isnan(tensor)
            if not nan_mask.all():  # 如果不是全部都是nan
                mean_val = tensor[~nan_mask].mean()
                tensor[nan_mask] = mean_val
            else:  # 如果全部都是nan，设置为1.0（对于scale）或0.0（对于zero）
                if 'scale' in tensor.name if hasattr(tensor, 'name') else 'scale':
                    tensor.fill_(1.0)
                else:
                    tensor.fill_(0.0)
        return tensor
    
    # 收集所有的scale和zero参数
    scale_params = {}
    zero_params = {}
    weight_params = {}
    
    # 遍历所有参数，分类收集
    for name, param in model.named_parameters():
        if '.scale' in name:
            base_name = name.replace('.scale', '')
            scale_params[base_name] = param
        elif '.zero' in name:
            base_name = name.replace('.zero', '')
            zero_params[base_name] = param
        elif '.weight' in name:
            base_name = name.replace('.weight', '')
            weight_params[base_name] = param
    
    # 处理每个权重层的量化参数
    for base_name, weight_param in weight_params.items():
        has_scale = base_name in scale_params
        has_zero = base_name in zero_params
        
        if has_scale:  # 如果存在scale参数
            with torch.no_grad():
                if has_zero:  # 非对称量化
                    print(f"Initializing asymmetric quantization for {base_name}")
                    scale_val, zero_val = asymmetric_quantization(weight_param.data)
                    
                    # 初始化scale
                    if scale_val.dim() == 0:  # 标量
                        scale_params[base_name].data.fill_(scale_val.item())
                    else:
                        scale_params[base_name].data.copy_(scale_val)
                    
                    # 初始化zero
                    if zero_val.dim() == 0:  # 标量
                        zero_params[base_name].data.fill_(zero_val.item())
                    else:
                        zero_params[base_name].data.copy_(zero_val)
                    
                    # 检查并替换nan值
                    scale_params[base_name].data = replace_nan_with_mean(scale_params[base_name].data)
                    zero_params[base_name].data = replace_nan_with_mean(zero_params[base_name].data)
                    
                else:  # 对称量化（只有scale，没有zero）
                    print(f"Initializing symmetric quantization for {base_name}")
                    scale_val = symmetric_quantization(weight_param.data)
                    
                    # 初始化scale
                    if scale_val.dim() == 0:  # 标量
                        scale_params[base_name].data.fill_(scale_val.item())
                    else:
                        scale_params[base_name].data.copy_(scale_val)
                    
                    # 检查并替换nan值
                    scale_params[base_name].data = replace_nan_with_mean(scale_params[base_name].data)
    
    print("Quantization parameters initialization completed!")
    return model


# 使用示例
if __name__ == "__main__":
    # 创建一个示例模型用于测试
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 5)
            # 模拟量化参数
            self.layer1_scale = nn.Parameter(torch.ones(5))
            self.layer1_zero = nn.Parameter(torch.zeros(5))
            
            self.layer2 = nn.Linear(5, 3)
            # 只有scale，没有zero（对称量化）
            self.layer2_scale = nn.Parameter(torch.ones(3))
    
        def forward(self, x):
            return self.layer2(self.layer1(x))
    
    # 测试函数
    model = TestModel()
    print("Before initialization:")
    print(f"layer1 scale: {model.layer1_scale.data}")
    print(f"layer1 zero: {model.layer1_zero.data}")
    print(f"layer2 scale: {model.layer2_scale.data}")
    
    model = initialize_quantization_params(model)
    
    print("\nAfter initialization:")
    print(f"layer1 scale: {model.layer1_scale.data}")
    print(f"layer1 zero: {model.layer1_zero.data}")
    print(f"layer2 scale: {model.layer2_scale.data}")