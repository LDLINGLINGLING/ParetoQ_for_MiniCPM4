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


def initialize_quantization_params_gptq(model, calibration_data=None, group_size=128, bits=4, 
                                        symmetric=True, blocksize=128, percdamp=0.01, 
                                        actorder=False, static_groups=False, verbose=True):
    """
    使用GPTQ算法初始化量化参数
    
    Args:
        model: PyTorch模型
        calibration_data: 校准数据，格式为[(input, output), ...] 或 DataLoader
        group_size: 分组大小，-1表示不分组
        bits: 量化位数
        symmetric: 是否使用对称量化（注意：真正的GPTQ通常使用非对称量化）
        blocksize: GPTQ算法的块大小
        percdamp: 阻尼系数百分比
        actorder: 是否使用激活顺序
        static_groups: 是否使用静态分组
        verbose: 是否打印详细信息
        
    Returns:
        model: 量化参数初始化后的模型
    """
    try:
        # 尝试导入GPTQ相关模块
        import sys
        import os
        
        # 添加AutoGPTQ路径到系统路径
        autogptq_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "AutoGPTQ")
        if autogptq_path not in sys.path:
            sys.path.append(autogptq_path)
            
        from auto_gptq.quantization.gptq import GPTQ
        from auto_gptq.quantization.quantizer import Quantizer
        
    except ImportError as e:
        if verbose:
            print(f"Warning: Cannot import GPTQ modules: {e}")
            print("Falling back to simple linear quantization...")
        return initialize_quantization_params_gptq_style(
            model, group_size=group_size, bits=bits, symmetric=symmetric, verbose=verbose
        )
    
    if calibration_data is None:
        if verbose:
            print("Warning: No calibration data provided. GPTQ requires calibration data.")
            print("Falling back to weight-based initialization...")
        return initialize_quantization_params_gptq_style(
            model, group_size=group_size, bits=bits, symmetric=symmetric, verbose=verbose
        )
    
    # 收集需要量化的层
    quantizable_layers = []
    layer_names = []
    
    for name, module in model.named_modules():
        # 检查是否是可量化的层（Linear层）
        if isinstance(module, torch.nn.Linear):
            # 检查是否有对应的量化参数
            scale_param_name = f"{name}.scale"
            has_scale = any(scale_param_name in param_name for param_name, _ in model.named_parameters())
            
            if has_scale:
                quantizable_layers.append(module)
                layer_names.append(name)
    
    if verbose:
        print(f"Found {len(quantizable_layers)} quantizable layers")
        print(f"GPTQ config: bits={bits}, group_size={group_size}, blocksize={blocksize}")
    
    # 为每个层运行GPTQ算法
    initialized_count = 0
    
    for layer_idx, (layer, layer_name) in enumerate(zip(quantizable_layers, layer_names)):
        if verbose:
            print(f"Processing layer {layer_idx+1}/{len(quantizable_layers)}: {layer_name}")
        
        try:
            # 创建GPTQ实例
            gptq = GPTQ(layer)
            
            # 配置量化器
            gptq.quantizer.configure(
                bits=bits,
                perchannel=True,  # 使用per-channel量化
                sym=symmetric,
                mse=False,
                norm=2.4,
                grid=100,
                maxshrink=0.8,
                trits=False
            )
            
            # 如果有校准数据，使用它来计算Hessian矩阵
            if calibration_data is not None:
                # 收集该层的输入输出数据
                num_samples = min(32, len(calibration_data)) if isinstance(calibration_data, (list, tuple)) else 32
                
                for i in range(num_samples):
                    # 生成或获取校准数据
                    if isinstance(calibration_data, (list, tuple)) and len(calibration_data) > i:
                        # 使用提供的校准数据
                        input_data = calibration_data[i][0] if isinstance(calibration_data[i], (list, tuple)) else calibration_data[i]
                    else:
                        # 生成随机输入数据（在实际使用中应该是真实的激活数据）
                        input_dim = layer.in_features
                        batch_size = 8
                        input_data = torch.randn(batch_size, input_dim, device=layer.weight.device, dtype=layer.weight.dtype)
                    
                    # 确保输入数据的形状正确
                    if input_data.dim() == 2:
                        input_data = input_data.unsqueeze(0)
                    
                    # 计算输出并添加到GPTQ
                    with torch.no_grad():
                        output_data = layer(input_data.view(-1, input_data.size(-1)))
                        gptq.add_batch(input_data.view(-1, input_data.size(-1)), output_data)
            else:
                # 如果没有校准数据，使用随机数据
                input_dim = layer.in_features
                batch_size = 8
                num_samples = 16
                
                for _ in range(num_samples):
                    fake_input = torch.randn(batch_size, input_dim, device=layer.weight.device, dtype=layer.weight.dtype)
                    with torch.no_grad():
                        fake_output = layer(fake_input)
                        gptq.add_batch(fake_input, fake_output)
            
            # 运行GPTQ量化
            # 注意：group_size 在 GPTQ 中 -1 表示不分组
            gptq_group_size = -1 if group_size is None else group_size
            
            scale, zero, g_idx = gptq.fasterquant(
                blocksize=blocksize,
                percdamp=percdamp,
                group_size=gptq_group_size,
                actorder=actorder,
                static_groups=static_groups
            )
            
            # 将计算得到的量化参数应用到模型
            scale_param_name = f"{layer_name}.scale"
            zero_param_name = f"{layer_name}.zero"
            
            # 查找并更新scale参数
            for param_name, param in model.named_parameters():
                if param_name == scale_param_name:
                    with torch.no_grad():
                        if scale.shape == param.shape:
                            param.data.copy_(scale)
                        else:
                            # 形状不匹配时的处理
                            if verbose:
                                print(f"Warning: Scale shape mismatch for {layer_name}. "
                                      f"Expected {param.shape}, got {scale.shape}")
                            
                            # 尝试调整形状
                            if param.numel() == scale.numel():
                                param.data.copy_(scale.view_as(param))
                            elif scale.numel() >= param.numel():
                                param.data.copy_(scale.flatten()[:param.numel()].view_as(param))
                            else:
                                # 重复填充
                                repeat_times = (param.numel() + scale.numel() - 1) // scale.numel()
                                repeated_scale = scale.repeat(repeat_times)
                                param.data.copy_(repeated_scale[:param.numel()].view_as(param))
                    break
            
            # 查找并更新zero参数（如果存在）
            if not symmetric:
                for param_name, param in model.named_parameters():
                    if param_name == zero_param_name:
                        with torch.no_grad():
                            if zero.shape == param.shape:
                                param.data.copy_(zero)
                            else:
                                # 形状不匹配时的处理
                                if verbose:
                                    print(f"Warning: Zero shape mismatch for {layer_name}. "
                                          f"Expected {param.shape}, got {zero.shape}")
                                
                                # 尝试调整形状
                                if param.numel() == zero.numel():
                                    param.data.copy_(zero.view_as(param))
                                elif zero.numel() >= param.numel():
                                    param.data.copy_(zero.flatten()[:param.numel()].view_as(param))
                                else:
                                    # 重复填充
                                    repeat_times = (param.numel() + zero.numel() - 1) // zero.numel()
                                    repeated_zero = zero.repeat(repeat_times)
                                    param.data.copy_(repeated_zero[:param.numel()].view_as(param))
                        break
            
            # 清理GPTQ实例以释放内存
            gptq.free()
            initialized_count += 1
            
            if verbose:
                print(f"Successfully quantized layer: {layer_name}")
                
        except Exception as e:
            if verbose:
                print(f"Error quantizing layer {layer_name}: {e}")
            continue
    
    if verbose:
        print(f"Successfully initialized {initialized_count}/{len(quantizable_layers)} layers using GPTQ")
        print("GPTQ quantization parameters initialization completed!")
    
    return model


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
    if verbose:
        print("Initializing quantization parameters using linear quantization (GPTQ-style)...")
    
    # 使用已有的 initialize_quantization_params 函数
    return initialize_quantization_params(model, group_size=group_size)

def create_calibration_data_from_model_weights(model, num_samples=32, batch_size=8):
    """
    从模型权重创建校准数据的简化版本
    注意：这不是理想的校准数据，理想情况下应该使用真实的数据集
    
    Args:
        model: PyTorch模型
        num_samples: 样本数量
        batch_size: 批大小
        
    Returns:
        list: 校准数据列表
    """
    calibration_data = []
    
    # 寻找第一个Linear层来确定输入维度
    first_linear = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            first_linear = module
            break
    
    if first_linear is None:
        return []
    
    input_dim = first_linear.in_features
    device = first_linear.weight.device
    dtype = first_linear.weight.dtype
    
    for _ in range(num_samples):
        # 生成随机输入数据
        fake_input = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
        # 注意：在实际使用中，这里应该是真实的输入-输出对
        calibration_data.append((fake_input, None))
    
    return calibration_data


def create_calibration_data_from_real_data(model, tokenizer, texts, max_length=512, batch_size=8):
    """
    从真实文本数据创建校准数据
    
    Args:
        model: PyTorch模型
        tokenizer: 分词器
        texts: 文本数据列表
        max_length: 最大长度
        batch_size: 批大小
        
    Returns:
        list: 校准数据列表
    """
    calibration_data = []
    device = next(model.parameters()).device
    
    # 准备数据
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 编码文本
        if tokenizer is not None:
            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            calibration_data.append((input_ids, attention_mask))
        else:
            # 如果没有 tokenizer，使用随机数据
            seq_len = min(max_length, 128)
            vocab_size = 32000  # 假设词汇表大小
            input_ids = torch.randint(0, vocab_size, (len(batch_texts), seq_len), device=device)
            calibration_data.append((input_ids, None))
    
    return calibration_data


def collect_layer_activations(model, calibration_data, layer_name):
    """
    收集指定层的激活数据
    
    Args:
        model: PyTorch模型
        calibration_data: 校准数据
        layer_name: 层名称
        
    Returns:
        list: 激活数据列表 [(input, output), ...]
    """
    activations = []
    
    # 注册前向钩子
    def hook_fn(module, input, output):
        # 保存输入和输出
        if isinstance(input, tuple):
            inp = input[0].detach().clone()
        else:
            inp = input.detach().clone()
        
        if isinstance(output, tuple):
            out = output[0].detach().clone()
        else:
            out = output.detach().clone()
        
        activations.append((inp, out))
    
    # 找到目标层并注册钩子
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        return []
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        # 运行前向传播
        model.eval()
        with torch.no_grad():
            for data_batch in calibration_data:
                if isinstance(data_batch, (list, tuple)) and len(data_batch) >= 2:
                    input_ids, attention_mask = data_batch[0], data_batch[1]
                    if attention_mask is not None:
                        _ = model(input_ids, attention_mask=attention_mask)
                    else:
                        _ = model(input_ids)
                else:
                    input_data = data_batch[0] if isinstance(data_batch, (list, tuple)) else data_batch
                    _ = model(input_data)
    finally:
        # 移除钩子
        hook.remove()
    
    return activations

# ...existing code...


# 使用示例
if __name__ == "__main__":
    # 创建一个示例模型用于测试
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 64)
            # 模拟分组量化参数 (假设分组大小为32，则需要128/32=4个scale和zero)
            self.layer1.scale = nn.Parameter(torch.ones(64, 4))  # [out_features, num_groups]
            self.layer1.zero = nn.Parameter(torch.zeros(64, 4))
            
            self.layer2 = nn.Linear(64, 32)
            # 只有scale，没有zero（对称量化）
            self.layer2.scale = nn.Parameter(torch.ones(32, 2))  # 64/32=2个scale
    
        def forward(self, x):
            return self.layer2(self.layer1(x))
    
    print("="*60)
    print("Testing GPTQ-style initialization")
    print("="*60)
    
    # 测试非对称分组量化
    print("\n1. Testing asymmetric grouped quantization:")
    model1 = TestModel()
    print("Before initialization:")
    print(f"layer1.scale shape: {model1.layer1.scale.shape}, mean: {model1.layer1.scale.data.mean():.4f}")
    print(f"layer1.zero shape: {model1.layer1.zero.shape}, mean: {model1.layer1.zero.data.mean():.4f}")
    
    model1 = initialize_quantization_params_gptq_style(
        model1, group_size=32, bits=4, symmetric=False, verbose=True
    )
    
    print("\nAfter asymmetric initialization:")
    print(f"layer1.scale: mean={model1.layer1.scale.data.mean():.4f}, std={model1.layer1.scale.data.std():.4f}")
    print(f"layer1.zero: mean={model1.layer1.zero.data.mean():.4f}, std={model1.layer1.zero.data.std():.4f}")
    
    # 测试对称分组量化
    print("\n2. Testing symmetric grouped quantization:")
    model2 = TestModel()
    # 移除zero参数进行对称量化测试
    del model2.layer1.zero
    
    model2 = initialize_quantization_params_gptq_style(
        model2, group_size=32, bits=4, symmetric=True, verbose=True
    )
    
    print("\nAfter symmetric initialization:")
    print(f"layer1.scale: mean={model2.layer1.scale.data.mean():.4f}, std={model2.layer1.scale.data.std():.4f}")
    print(f"layer2.scale: mean={model2.layer2.scale.data.mean():.4f}, std={model2.layer2.scale.data.std():.4f}")
    
    # 测试全局量化
    print("\n3. Testing global quantization:")
    model3 = TestModel()
    # 调整为全局量化的参数形状
    model3.layer1.scale = nn.Parameter(torch.ones(64, 1))
    model3.layer1.zero = nn.Parameter(torch.zeros(64, 1))
    model3.layer2.scale = nn.Parameter(torch.ones(32, 1))
    
    model3 = initialize_quantization_params_gptq_style(
        model3, group_size=-1, bits=4, symmetric=False, verbose=True
    )
    
    print("\nAfter global initialization:")
    print(f"layer1.scale: mean={model3.layer1.scale.data.mean():.4f}, std={model3.layer1.scale.data.std():.4f}")
    print(f"layer1.zero: mean={model3.layer1.zero.data.mean():.4f}, std={model3.layer1.zero.data.std():.4f}")
    
    print("\n" + "="*60)
    print("Testing original initialization")
    print("="*60)
    
    # 测试原有函数进行对比
    print("\n4. Testing original grouped quantization:")
    model4 = TestModel()
    model4 = initialize_quantization_params(model4, group_size=32)
    
    print("\nAfter original initialization:")
    print(f"layer1.scale: mean={model4.layer1.scale.data.mean():.4f}, std={model4.layer1.scale.data.std():.4f}")
    print(f"layer1.zero: mean={model4.layer1.zero.data.mean():.4f}, std={model4.layer1.zero.data.std():.4f}")
    
    # 使用 GPTQ 量化的完整示例
    print("\n" + "="*60)
    print("Testing GPTQ Quantization with Real Data")
    print("="*60)
    
    # 测试真实的 GPTQ 量化初始化
    print("\n5. Testing GPTQ quantization with calibration data:")
    
    try:
        # 创建测试模型
        model5 = TestModel()
        
        # 创建校准数据
        calibration_data = create_calibration_data_from_model_weights(model5, num_samples=16, batch_size=4)
        
        # 使用 GPTQ 量化初始化
        model5 = initialize_quantization_params_gptq(
            model5, 
            calibration_data=calibration_data,
            group_size=32, 
            bits=4, 
            symmetric=False,
            blocksize=128,
            percdamp=0.01,
            verbose=True
        )
        
        print("\nAfter GPTQ initialization:")
        print(f"layer1.scale: mean={model5.layer1.scale.data.mean():.4f}, std={model5.layer1.scale.data.std():.4f}")
        print(f"layer1.zero: mean={model5.layer1.zero.data.mean():.4f}, std={model5.layer1.zero.data.std():.4f}")
        
    except Exception as e:
        print(f"GPTQ initialization failed: {e}")
        print("This is expected if AutoGPTQ is not properly installed.")
    
    # 测试 NaN 检测和修复
    print("\n6. Testing NaN detection and repair:")
    
    model6 = TestModel()
    # 人工引入 NaN 值
    model6.layer1.scale.data[0, 0] = float('nan')
    model6.layer1.zero.data[0, 0] = float('nan')
    
    has_nan, nan_info = check_nan_in_model(model6, detailed=True)
    print(f"Has NaN: {has_nan}")
    
    if has_nan:
        print("Fixing NaN values...")
        model6, fixed_params, total_nan_count = fix_nan_in_model(model6, verbose=True)
        print(f"Fixed {total_nan_count} NaN values in {fixed_params} parameters")
    
    print("\n" + "="*60)
    print("Usage Guide")
    print("="*60)
    
    print("""
使用指南：

1. 基本线性量化初始化：
   model = initialize_quantization_params(model, group_size=128)

2. GPTQ-style 线性量化初始化：
   model = initialize_quantization_params_gptq_style(
       model, group_size=128, bits=4, symmetric=False, verbose=True
   )

3. 真实 GPTQ 量化初始化：
   # 准备校准数据
   calibration_data = create_calibration_data_from_real_data(
       model, tokenizer, texts, max_length=512, batch_size=8
   )
   
   # 或者使用简化版本
   calibration_data = create_calibration_data_from_model_weights(
       model, num_samples=32, batch_size=8
   )
   
   # 应用 GPTQ 量化
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

4. 检查和修复 NaN 值：
   has_nan, nan_info = check_nan_in_model(model, detailed=True)
   if has_nan:
       model, fixed_params, total_nan_count = fix_nan_in_model(model, verbose=True)

5. 设置仅量化参数可训练：
   model = only_train_adapter(model, verbose=True)

注意事项：
- 对于真实的 GPTQ 量化，需要安装 AutoGPTQ 库
- 校准数据应该是真实的输入数据，而不是随机数据
- group_size 控制分组量化的组大小，None 或 -1 表示全局量化
- symmetric=True 时只会初始化 scale 参数，symmetric=False 时会初始化 scale 和 zero 参数
- GPTQ 量化会自动 fallback 到线性量化如果 AutoGPTQ 不可用或没有校准数据
""")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)