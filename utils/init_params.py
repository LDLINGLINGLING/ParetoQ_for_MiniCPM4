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
            weight_grouped = weight_flat.reshape(-1, group_size).float()
            
            # 对每组计算min和max
            min_vals =  torch.quantile(weight_grouped , 0.01,dim=-1)[0]
            max_vals = torch.quantile(weight_grouped , 0.99,dim=-1)[0]
            
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
def load_gptq_weights_only(model_path=None, verbose=True):
    """
    只读取GPTQ权重，返回字典（层名->权重tensor）

    Args:
        model_path: 权重文件路径（.bin或.safetensors），为目录时自动查找权重文件
        verbose: 是否打印详细信息

    Returns:
        weights_dict: {层名: 权重tensor}
    """
    import os
    import torch

    # 默认模型路径
    if model_path is None:
        model_path = r"D:\model_best\minicpm\pretrain_model\Qwen3-0___6B-GPTQ-Int8"

    # 如果是目录，自动查找权重文件
    if os.path.isdir(model_path):
        # 优先找safetensors
        files = os.listdir(model_path)
        weight_file = None
        for f in files:
            if f.endswith('.safetensors'):
                weight_file = os.path.join(model_path, f)
                break
        if weight_file is None:
            for f in files:
                if f.endswith('.bin') or f.endswith('.pt'):
                    weight_file = os.path.join(model_path, f)
                    break
        if weight_file is None:
            raise FileNotFoundError("未找到权重文件（.safetensors/.bin/.pt）")
    else:
        weight_file = model_path

    if verbose:
        print(f"Loading weights from: {weight_file}")

    # 加载权重
    if weight_file.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            weights_dict = {}
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights_dict[key] = f.get_tensor(key)
        except ImportError:
            raise ImportError("pip install safetensors")
    else:
        weights_dict = torch.load(weight_file, map_location='cpu')
        # 可能是state_dict或直接是权重字典
        if "state_dict" in weights_dict:
            weights_dict = weights_dict["state_dict"]

    if verbose:
        print(f"Loaded {len(weights_dict)} tensors.")

    return weights_dict

def initialize_from_gptq_model(model, gptq_weights, verbose=True):
    """
    从GPTQ权重中提取scale和zero参数，初始化QAT模型的量化参数
    
    Args:
        model: 目标QAT模型
        gptq_weights: GPTQ权重字典
        verbose: 是否打印详细信息
        
    Returns:
        model: 初始化后的模型
    """
    import torch
    import numpy as np
    
    def unpack_gptq_zeros(qzeros, bits, group_size, outfeatures):
        """解包GPTQ的qzeros到原始zeros"""
        # qzeros shape: [num_groups, outfeatures // 32 * bits]
        # 需要解包成: [num_groups, outfeatures]
        
        zeros = torch.zeros((qzeros.shape[0], outfeatures), dtype=torch.float32)
        
        if bits in [2, 4, 8]:
            # 对于2,4,8位量化，复制qlinear_cuda.py中的解包逻辑
            wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32)
            
            # 解包过程（完全按照qlinear_cuda.py的forward方法）
            unpacked = torch.bitwise_right_shift(
                torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
                wf.unsqueeze(0).unsqueeze(0)
            ).to(torch.int16 if bits == 8 else torch.int8)
            unpacked = torch.bitwise_and(unpacked, (2**bits) - 1)
            
            # 重塑为目标形状
            unpacked = unpacked.reshape(qzeros.shape[0], -1)
            
            # 确保不超出outfeatures维度
            copy_cols = min(unpacked.shape[1], outfeatures)
            zeros[:, :copy_cols] = unpacked[:, :copy_cols].float()
            
        elif bits == 3:
            # 3位量化的特殊处理（按照qlinear_cuda.py的逻辑）
            # 重塑qzeros为3D形状处理
            qzeros_reshaped = qzeros.reshape(qzeros.shape[0], qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
            
            # 定义wf用于3位解包
            wf = torch.tensor([
                [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31], 
                [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
            ], dtype=torch.int32).reshape(1, 3, 12)
            
            # 解包过程
            unpacked = qzeros_reshaped >> wf.unsqueeze(0)
            unpacked[:, :, 0, 10] = (unpacked[:, :, 0, 10] & 0x3) | ((unpacked[:, :, 1, 0] << 2) & 0x4)
            unpacked[:, :, 1, 11] = (unpacked[:, :, 1, 11] & 0x1) | ((unpacked[:, :, 2, 0] << 1) & 0x6)
            unpacked = unpacked & 0x7
            
            # 拼接结果
            unpacked = torch.cat([
                unpacked[:, :, 0, :11], 
                unpacked[:, :, 1, 1:12], 
                unpacked[:, :, 2, 1:11]
            ], dim=2)
            
            # 重塑为最终形状
            unpacked = unpacked.reshape(qzeros.shape[0], -1)
            copy_cols = min(unpacked.shape[1], outfeatures)
            zeros[:, :copy_cols] = unpacked[:, :copy_cols].float()
        
        # 重要：加1恢复原始零点值（因为GPTQ在pack时减了1）
        zeros = zeros + 1
        return zeros
    
    def get_bits_from_weight_shape(qweight_shape, weight_shape):
        """从qweight形状推断量化位数"""
        # qweight shape: [infeatures // 32 * bits, outfeatures]
        # weight shape: [outfeatures, infeatures] 
        infeatures = weight_shape[1]
        qweight_rows = qweight_shape[0]
        
        # 计算bits: qweight_rows = infeatures // 32 * bits
        # 所以 bits = qweight_rows * 32 / infeatures
        bits = int(qweight_rows * 32 / infeatures)
        return bits
    
    def calculate_group_size(gptq_scales, gptq_weight):
        """计算GPTQ模型中实际使用的分组大小"""
        # gptq_scales shape: [num_groups, outfeatures]
        # gptq_weight shape: [outfeatures, infeatures]
        num_groups = gptq_scales.shape[0]
        infeatures = gptq_weight.shape[1]
        
        # 分组大小 = infeatures / num_groups
        group_size = infeatures // num_groups
        
        if verbose:
            print(f"  - Detected group_size: {group_size} (infeatures: {infeatures}, num_groups: {num_groups})")
        
        return group_size
    
    initialized_layers = 0
    
    # 遍历模型中的所有参数
    for name, param in model.named_parameters():
        if '.scale' in name or '.zero' in name:
            # 获取基础层名
            if '.scale' in name:
                base_name = name.replace('.scale', '')
                param_type = 'scale'
            else:
                base_name = name.replace('.zero', '')
                param_type = 'zero'
            
            # 在GPTQ权重中查找对应的参数
            gptq_scales_key = base_name + '.scales'
            gptq_qzeros_key = base_name + '.qzeros'
            gptq_qweight_key = base_name + '.qweight'
            gptq_weight_key = base_name + '.weight'
            
            if gptq_scales_key in gptq_weights and gptq_qzeros_key in gptq_weights:
                with torch.no_grad():
                    gptq_scales = gptq_weights[gptq_scales_key]
                    gptq_qzeros = gptq_weights[gptq_qzeros_key] 
                    
                    # 获取weight信息用于计算分组
                    if gptq_weight_key in gptq_weights:
                        gptq_weight = gptq_weights[gptq_weight_key]
                        group_size = calculate_group_size(gptq_scales, gptq_weight)
                    else:
                        # 从scales形状推断
                        group_size = 128  # 默认值
                        if verbose:
                            print(f"⚠ Cannot find weight for {base_name}, using default group_size={group_size}")
                    
                    if param_type == 'scale':
                        # 直接使用GPTQ的scales
                        # gptq_scales shape: [num_groups, outfeatures]
                        # 需要转置到 [outfeatures, num_groups]
                        target_scales = gptq_scales.t().contiguous()
                        
                        # 确保数据类型匹配
                        target_scales = target_scales.to(param.dtype)
                        
                        # 检查形状是否匹配
                        if target_scales.shape == param.shape:
                            param.data.copy_(target_scales)
                            if verbose:
                                print(f"✓ Initialized {name} from GPTQ scales, shape: {param.shape}")
                        else:
                            if verbose:
                                print(f"⚠ Shape mismatch for {name}: target {param.shape} vs GPTQ {target_scales.shape}")
                            # 尝试重塑或裁剪
                            if target_scales.numel() >= param.numel():
                                param.data.copy_(target_scales.flatten()[:param.numel()].view_as(param.data))
                            else:
                                # 重复填充
                                repeated = target_scales.flatten().repeat(param.numel() // target_scales.numel() + 1)
                                param.data.copy_(repeated[:param.numel()].view_as(param.data))
                            if verbose:
                                print(f"✓ Reshaped and initialized {name}")
                    
                    elif param_type == 'zero':
                        # 需要解包qzeros
                        # 首先确定量化位数
                        if gptq_qweight_key in gptq_weights:
                            gptq_qweight = gptq_weights[gptq_qweight_key]
                            if gptq_weight_key in gptq_weights:
                                bits = get_bits_from_weight_shape(gptq_qweight.shape, gptq_weight.shape)
                            else:
                                # 从qzeros形状推断bits
                                # qzeros shape: [num_groups, outfeatures // 32 * bits]
                                # outfeatures 从 scales 获取
                                outfeatures = gptq_scales.shape[1]
                                expected_cols = outfeatures // 32  # 假设至少4位
                                actual_cols = gptq_qzeros.shape[1]
                                bits = int(actual_cols * 32 / outfeatures)
                                if verbose:
                                    print(f"  - Inferred bits from qzeros shape: {bits}")
                        else:
                            # 默认使用4位
                            bits = 4
                            if verbose:
                                print(f"⚠ Cannot determine bits for {name}, using default 4-bit")
                        
                        # 解包zeros
                        outfeatures = gptq_scales.shape[1]  # scales shape: [num_groups, outfeatures]
                        
                        unpacked_zeros = unpack_gptq_zeros(gptq_qzeros, bits, group_size, outfeatures)
                        
                        # 转置到目标形状 [outfeatures, num_groups]
                        target_zeros = unpacked_zeros.t().contiguous()
                        
                        # 确保数据类型匹配
                        target_zeros = target_zeros.to(param.dtype)
                        
                        # 检查形状是否匹配
                        if target_zeros.shape == param.shape:
                            param.data.copy_(target_zeros)
                            if verbose:
                                print(f"✓ Initialized {name} from GPTQ qzeros, shape: {param.shape}")
                        else:
                            if verbose:
                                print(f"⚠ Shape mismatch for {name}: target {param.shape} vs GPTQ {target_zeros.shape}")
                            # 尝试重塑或裁剪
                            if target_zeros.numel() >= param.numel():
                                param.data.copy_(target_zeros.flatten()[:param.numel()].view_as(param.data))
                            else:
                                # 重复填充
                                repeated = target_zeros.flatten().repeat(param.numel() // target_zeros.numel() + 1)
                                param.data.copy_(repeated[:param.numel()].view_as(param.data))
                            if verbose:
                                print(f"✓ Reshaped and initialized {name}")
                    
                    initialized_layers += 1
    
    if verbose:
        print(f"\n=== GPTQ Initialization Summary ===")
        print(f"✓ Initialized {initialized_layers} quantization parameters from GPTQ weights")
        print(f"✓ All scale and zero parameters have been updated")
    
    return model


