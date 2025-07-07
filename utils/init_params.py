import torch
import torch.nn as nn
import re

def only_train_adapter(model, verbose=True):
    for name, param in model.named_parameters():
        if 'zero' in name.lower() or 'scale' in name.lower():
            if verbose:
                print(f"Train parameter: {name}")
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
def fix_nan_in_model(model, verbose=True, inplace=True):
    """
    自动修复模型中包含NaN值的参数 - 仅修改NaN位置
         
    Args:
        model: 要修复的PyTorch模型
        verbose: 是否打印修复信息
        inplace: 是否原地修改模型，如果False则返回修复后的新模型
    
    Returns:
        model: 修复后的模型
        fixed_params: 修复的参数数量
        total_nan_count: 总共修复的NaN值数量
    """
    import torch
    import copy
    
    # 如果不是原地修改，创建模型的深拷贝
    if not inplace:
        model = copy.deepcopy(model)
    
    fixed_params = 0
    total_nan_count = 0
         
    for name, param in model.named_parameters():
        if param.is_meta:
            continue  # 跳过meta tensor
                     
        # 检查是否包含NaN
        nan_mask = torch.isnan(param)
        # if not nan_mask.any():
        #     continue  # 没有NaN则跳过
        
        # 统计NaN数量
        nan_count = nan_mask.sum().item()
        total_nan_count += nan_count
                     
        # 获取设备和数据类型信息
        device = param.device
        dtype = param.dtype
        
        # 根据参数名决定替换策略
        with torch.no_grad():  # 确保不影响梯度计算
            if 'scale' in name.lower():
                # 整个scale参数矩阵替换为1
                param.data = torch.ones_like(param, device=device, dtype=dtype)
                if verbose:
                    print(f"Fully reset [scale] param {name} -> all 1.0 (shape {param.shape})")
                    
            elif 'zero' in name.lower() or 'bias' in name.lower():
                # 整个zero/bias参数矩阵替换为0
                param.data = torch.zeros_like(param, device=device, dtype=dtype)
                if verbose:
                    print(f"Fully reset [zero/bias] param {name} -> all 0.0 (shape {param.shape})")
                         
            
        
        fixed_params += 1

    if verbose:
        print(f"\nSummary:")
        print(f"- Fixed parameters: {fixed_params}")
        print(f"- Total NaN values fixed: {total_nan_count}")
        print(f"- Model modification: {'In-place' if inplace else 'New copy'}")
        
    return model, fixed_params, total_nan_count


def check_nan_in_model(model, detailed=False):
    """
    检查模型中的NaN值
    
    Args:
        model: PyTorch模型
        detailed: 是否显示详细信息
    
    Returns:
        has_nan: 是否包含NaN
        nan_info: NaN信息字典
    """
    import torch
    
    nan_info = {}
    total_nan_count = 0
    
    for name, param in model.named_parameters():
        if param.is_meta:
            continue
            
        nan_mask = torch.isnan(param)
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            total_nan_count += nan_count
            nan_info[name] = {
                'count': nan_count,
                'total_elements': param.numel(),
                'percentage': (nan_count / param.numel()) * 100,
                'shape': param.shape
            }
            
            if detailed:
                print(f"Parameter: {name}")
                print(f"  - Shape: {param.shape}")
                print(f"  - NaN count: {nan_count}/{param.numel()} ({nan_count/param.numel()*100:.2f}%)")
                
                # 显示NaN位置的示例（如果不是太多）
                if nan_count <= 10:
                    nan_indices = torch.nonzero(nan_mask)
                    print(f"  - NaN positions: {nan_indices.tolist()}")
                print()
    
    has_nan = total_nan_count > 0
    
    if detailed:
        if has_nan:
            print(f"Total NaN values found: {total_nan_count}")
        else:
            print("No NaN values found in the model.")
    
    return has_nan, nan_info
import torch
import torch.nn as nn
import re

def initialize_quantization_params(model, group_size=None):
    """
    初始化模型中的量化参数（scale和zero），支持分组量化
    
    Args:
        model: PyTorch模型
        group_size: 分组大小，如果为None则使用全局量化
        
    Returns:
        model: 初始化后的模型
    """
    
    def get_weight_param_name(scale_name):
        """从scale参数名获取对应的weight参数名"""
        # 将.scale替换为.weight，或将.zero替换为.weight
        weight_name = scale_name.replace('.scale', '.weight').replace('.zero', '.weight')
        return weight_name
    
    def symmetric_quantization_grouped(weight, group_size=None, bits=8):
        """分组对称量化：计算scale"""
        if group_size is None:
            # 全局量化
            max_val = torch.max(torch.abs(weight))
            scale = max_val / (2**(bits-1) - 1)
            return scale
        else:
            # 分组量化
            # 将权重重塑为可以分组的形状
            original_shape = weight.shape
            weight_flat = weight.flatten()
            
            # 确保能够整除，不够的话用0填充
            total_elements = weight_flat.numel()
            if total_elements % group_size != 0:
                pad_size = group_size - (total_elements % group_size)
                weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, dtype=weight.dtype, device=weight.device)])
            
            # 重塑为分组形状 [num_groups, group_size]
            weight_grouped = weight_flat.reshape(-1, group_size)
            
            # 对每组计算scale
            max_vals = torch.max(torch.abs(weight_grouped), dim=1)[0]
            scales = max_vals / (2**(bits-1) - 1)
            
            # 处理可能的零值或极小值
            scales = torch.clamp(scales, min=1e-8)
            
            return scales
    
    def asymmetric_quantization_grouped(weight, group_size=None, bits=8):
        """分组非对称量化：计算scale和zero"""
        if group_size is None:
            # 全局量化
            min_val = torch.min(weight)
            max_val = torch.max(weight)
            
            scale = (max_val - min_val) / (2**bits - 1)
            zero = torch.round(-min_val / scale)
            
            return scale, zero
        else:
            # 分组量化
            # 将权重重塑为可以分组的形状
            original_shape = weight.shape
            weight_flat = weight.flatten()
            
            # 确保能够整除，不够的话用0填充
            total_elements = weight_flat.numel()
            if total_elements % group_size != 0:
                pad_size = group_size - (total_elements % group_size)
                weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, dtype=weight.dtype, device=weight.device)])
            
            # 重塑为分组形状 [num_groups, group_size]
            weight_grouped = weight_flat.reshape(-1, group_size)
            
            # 对每组计算min和max
            min_vals = torch.min(weight_grouped, dim=1)[0]
            max_vals = torch.max(weight_grouped, dim=1)[0]
            
            # 计算scale和zero
            scales = (max_vals - min_vals) / (2**bits - 1)
            zeros = torch.round(-min_vals / scales)
            
            # 处理可能的零值或极小值
            scales = torch.clamp(scales, min=1e-8)
            
            return scales, zeros
    
    def replace_nan_with_mean(tensor):
        """将tensor中的nan值替换为非nan值的平均值"""
        if torch.isnan(tensor).any():
            nan_mask = torch.isnan(tensor)
            if not nan_mask.all():  # 如果不是全部都是nan
                mean_val = tensor[~nan_mask].mean()
                tensor[nan_mask] = mean_val
            else:  # 如果全部都是nan，设置默认值
                if tensor.numel() > 1:  # 多元素tensor
                    tensor.fill_(1.0)  # 对于scale设置为1.0
                else:  # 单元素tensor
                    tensor.fill_(1.0)
        return tensor
    
    def replace_nan_with_mean_zero(tensor):
        """将zero tensor中的nan值替换为非nan值的平均值"""
        if torch.isnan(tensor).any():
            nan_mask = torch.isnan(tensor)
            if not nan_mask.all():  # 如果不是全部都是nan
                mean_val = tensor[~nan_mask].mean()
                tensor[nan_mask] = mean_val
            else:  # 如果全部都是nan，设置为0.0
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
                    if group_size is None:
                        print(f"Initializing global asymmetric quantization for {base_name}")
                    else:
                        print(f"Initializing grouped asymmetric quantization for {base_name} (group_size={group_size})")
                    
                    scale_val, zero_val = asymmetric_quantization_grouped(weight_param.data, group_size)
                    
                    # 初始化scale
                    if scale_val.dim() == 0:  # 标量
                        if scale_params[base_name].numel() == 1:
                            scale_params[base_name].data.fill_(scale_val.item())
                        else:
                            scale_params[base_name].data.fill_(scale_val.item())
                    else:
                        # 确保scale参数的形状匹配
                        if scale_params[base_name].numel() == scale_val.numel():
                            scale_params[base_name].data.copy_(scale_val.view_as(scale_params[base_name].data))
                        else:
                            print(f"Warning: Scale parameter shape mismatch for {base_name}. "
                                  f"Expected {scale_params[base_name].shape}, got {scale_val.shape}")
                            # 尝试重塑或截断
                            if scale_val.numel() >= scale_params[base_name].numel():
                                scale_params[base_name].data.copy_(scale_val.flatten()[:scale_params[base_name].numel()].view_as(scale_params[base_name].data))
                            else:
                                # 如果计算出的scale太少，则重复填充
                                repeated_scale = scale_val.repeat(scale_params[base_name].numel() // scale_val.numel() + 1)
                                scale_params[base_name].data.copy_(repeated_scale[:scale_params[base_name].numel()].view_as(scale_params[base_name].data))
                    
                    # 初始化zero
                    if zero_val.dim() == 0:  # 标量
                        if zero_params[base_name].numel() == 1:
                            zero_params[base_name].data.fill_(zero_val.item())
                        else:
                            zero_params[base_name].data.fill_(zero_val.item())
                    else:
                        # 确保zero参数的形状匹配
                        if zero_params[base_name].numel() == zero_val.numel():
                            zero_params[base_name].data.copy_(zero_val.view_as(zero_params[base_name].data))
                        else:
                            print(f"Warning: Zero parameter shape mismatch for {base_name}. "
                                  f"Expected {zero_params[base_name].shape}, got {zero_val.shape}")
                            # 尝试重塑或截断
                            if zero_val.numel() >= zero_params[base_name].numel():
                                zero_params[base_name].data.copy_(zero_val.flatten()[:zero_params[base_name].numel()].view_as(zero_params[base_name].data))
                            else:
                                # 如果计算出的zero太少，则重复填充
                                repeated_zero = zero_val.repeat(zero_params[base_name].numel() // zero_val.numel() + 1)
                                zero_params[base_name].data.copy_(repeated_zero[:zero_params[base_name].numel()].view_as(zero_params[base_name].data))
                    
                    # 检查并替换nan值
                    scale_params[base_name].data = replace_nan_with_mean(scale_params[base_name].data)
                    zero_params[base_name].data = replace_nan_with_mean_zero(zero_params[base_name].data)
                    
                else:  # 对称量化（只有scale，没有zero）
                    if group_size is None:
                        print(f"Initializing global symmetric quantization for {base_name}")
                    else:
                        print(f"Initializing grouped symmetric quantization for {base_name} (group_size={group_size})")
                    
                    scale_val = symmetric_quantization_grouped(weight_param.data, group_size)
                    
                    # 初始化scale
                    if scale_val.dim() == 0:  # 标量
                        if scale_params[base_name].numel() == 1:
                            scale_params[base_name].data.fill_(scale_val.item())
                        else:
                            scale_params[base_name].data.fill_(scale_val.item())
                    else:
                        # 确保scale参数的形状匹配
                        if scale_params[base_name].numel() == scale_val.numel():
                            scale_params[base_name].data.copy_(scale_val.view_as(scale_params[base_name].data))
                        else:
                            print(f"Warning: Scale parameter shape mismatch for {base_name}. "
                                  f"Expected {scale_params[base_name].shape}, got {scale_val.shape}")
                            # 尝试重塑或截断
                            if scale_val.numel() >= scale_params[base_name].numel():
                                scale_params[base_name].data.copy_(scale_val.flatten()[:scale_params[base_name].numel()].view_as(scale_params[base_name].data))
                            else:
                                # 如果计算出的scale太少，则重复填充
                                repeated_scale = scale_val.repeat(scale_params[base_name].numel() // scale_val.numel() + 1)
                                scale_params[base_name].data.copy_(repeated_scale[:scale_params[base_name].numel()].view_as(scale_params[base_name].data))
                    
                    # 检查并替换nan值
                    scale_params[base_name].data = replace_nan_with_mean(scale_params[base_name].data)
    
    if group_size is None:
        print("Global quantization parameters initialization completed!")
    else:
        print(f"Grouped quantization parameters initialization completed! (group_size={group_size})")
    
    return model


# 使用示例
if __name__ == "__main__":
    # 创建一个示例模型用于测试
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 64)
            # 模拟分组量化参数 (假设分组大小为32，则需要128/32=4个scale和zero)
            self.layer1_scale = nn.Parameter(torch.ones(4))
            self.layer1_zero = nn.Parameter(torch.zeros(4))
            
            self.layer2 = nn.Linear(64, 32)
            # 只有scale，没有zero（对称量化）
            self.layer2_scale = nn.Parameter(torch.ones(2))  # 64/32=2个scale
    
        def forward(self, x):
            return self.layer2(self.layer1(x))
    
    # 测试函数
    model = TestModel()
    print("Before initialization:")
    print(f"layer1 scale: {model.layer1_scale.data}")
    print(f"layer1 zero: {model.layer1_zero.data}")
    print(f"layer2 scale: {model.layer2_scale.data}")
    
    # 测试分组量化
    print("\n" + "="*50)
    print("Testing grouped quantization:")
    model = initialize_quantization_params(model, group_size=32)
    
    print("\nAfter grouped initialization:")
    print(f"layer1 scale: {model.layer1_scale.data}")
    print(f"layer1 zero: {model.layer1_zero.data}")
    print(f"layer2 scale: {model.layer2_scale.data}")
    
    # 测试全局量化
    print("\n" + "="*50)
    print("Testing global quantization:")
    model2 = TestModel()
    # 全局量化时scale应该是标量
    model2.layer1_scale = nn.Parameter(torch.ones(1))
    model2.layer1_zero = nn.Parameter(torch.zeros(1))
    model2.layer2_scale = nn.Parameter(torch.ones(1))
    
    model2 = initialize_quantization_params(model2, group_size=None)
    
    print("\nAfter global initialization:")
    print(f"layer1 scale: {model2.layer1_scale.data}")
    print(f"layer1 zero: {model2.layer1_zero.data}")
    print(f"layer2 scale: {model2.layer2_scale.data}")