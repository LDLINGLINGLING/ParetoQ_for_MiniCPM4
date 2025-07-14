"""
基于Straight-Through Estimator的LSQ GroupWise扩展
采用STE方式处理round和clamp操作的梯度，保持与utils_group_zero_min.py一致的代码结构
"""

import torch
import torch.nn.functional as F
import math
import torch.nn as nn

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min_val, max_val):
    """
    Implement Straight-Through Estimator for clamp operation.
    """
    return (x.clamp(min_val, max_val) - x).detach() + x

def _reshape_for_groupwise(input_tensor, alpha, group_size, dim):
    """
    采用与utils_group_zero_min.py相同的分组重塑方式
    For QAT: input is weight tensor [out_features, in_features], 
    alpha is [out_features, num_groups] where num_groups = ceil(in_features / group_size)
    
    :param input_tensor: input tensor to be reshaped [out_features, in_features]
    :param alpha: alpha tensor (per group) [out_features, num_groups]
    :param group_size: size of each group
    :param dim: dimension along which to group (should be -1 for in_features)
    :return: reshaped input and expanded alpha
    """
    # Handle negative dimension
    if dim < 0:
        dim = input_tensor.dim() + dim
    
    # For QAT weight quantization, we expect:
    # input_tensor: [out_features, in_features]
    # alpha: [out_features, num_groups]
    # dim: -1 (grouping along in_features dimension)
    
    if input_tensor.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {input_tensor.dim()}D tensor")
    
    if dim != 1:  # dim=1 corresponds to in_features dimension
        raise ValueError(f"For weight quantization, grouping should be along in_features dimension (dim=1), got dim={dim}")
    
    out_features, in_features = input_tensor.shape
    num_groups = (in_features + group_size - 1) // group_size
    
    # Verify alpha shape
    expected_alpha_shape = (out_features, num_groups)
    if alpha.shape != expected_alpha_shape:
        raise ValueError(f"Expected alpha shape {expected_alpha_shape}, got {alpha.shape}")
    
    # Pad in_features dimension if necessary
    if in_features % group_size != 0:
        pad_size = num_groups * group_size - in_features
        # Pad on the right side of in_features dimension
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_size))
    
    # Reshape input: [out_features, in_features] -> [out_features, num_groups, group_size]
    input_reshaped = input_tensor.view(out_features, num_groups, group_size)
    
    # Expand alpha: [out_features, num_groups] -> [out_features, num_groups, group_size]
    alpha_expanded = alpha.unsqueeze(-1).expand(out_features, num_groups, group_size)
    
    return input_reshaped, alpha_expanded

def _sum_over_groups(grad_tensor, group_size, dim, alpha_shape):
    """
    采用与utils_group_zero_min.py相同的分组求和方式
    Sum gradients over groups to get per-group gradients.
    For QAT: grad_tensor has shape [out_features, num_groups, group_size]
    Need to sum over group_size dimension to get [out_features, num_groups]
    
    :param grad_tensor: gradient tensor [out_features, num_groups, group_size]
    :param group_size: size of each group
    :param dim: dimension along which groups are organized (should be 1 for in_features)
    :param alpha_shape: target shape for alpha gradients [out_features, num_groups]
    :return: summed gradients per group [out_features, num_groups]
    """
    # For the corrected reshape, grad_tensor has shape [out_features, num_groups, group_size]
    # We need to sum over the group_size dimension (last dimension)
    grad_alpha = grad_tensor.sum(dim=-1)  # Sum over group_size dimension
    
    # grad_alpha now has shape [out_features, num_groups] which matches alpha_shape
    return grad_alpha

def lsq_groupwise_quantize(input, alpha, num_bits, group_size, dim=-1, zero_point=None):
    """
    Group-wise quantization using STE approach.
    This function is directly differentiable without custom autograd.Function.
    
    :param input: input to be quantized
    :param alpha: the step size (per group)
    :param num_bits: quantization bits
    :param group_size: size of each group for quantization
    :param dim: dimension along which to group (default: -1, last dimension)
    :param zero_point: zero point for asymmetric quantization (per group), None for symmetric
    :return: quantized output
    """
    if num_bits >= 16:
        return input

    # Calculate quantization range
    if zero_point is not None:
        # Asymmetric quantization: 0 to 2^num_bits - 1
        if num_bits == 1:
            Qn, Qp = 0, 1
        else:
            Qn, Qp = 0, 2 ** num_bits - 1
    else:
        # Symmetric quantization: -2^(num_bits-1) to 2^(num_bits-1) - 1
        if num_bits == 1 or num_bits == 0:
            Qn, Qp = -1, 1
        else:
            Qn, Qp = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1

    # Increase minimum alpha value to prevent numerical instability
    eps = torch.tensor(1e-5, device=alpha.device, dtype=alpha.dtype)
    alpha = torch.clamp(alpha, min=eps)
    
    # Check for NaN/Inf in alpha
    if torch.any(torch.isnan(alpha)) or torch.any(torch.isinf(alpha)):
        print("Warning: NaN or Inf detected in alpha during forward pass")
        alpha = torch.where(torch.isnan(alpha) | torch.isinf(alpha), eps, alpha)

    # Reshape input and alpha for group-wise processing
    original_shape = input.shape
    input_reshaped, alpha_expanded = _reshape_for_groupwise(input, alpha, group_size, dim)

    # Handle zero point for asymmetric quantization
    zero_point_expanded = None
    if zero_point is not None:
        # 使用传入的zero_point参数，而不是计算每组的最小值
        zero_point_reshaped, zero_point_expanded = _reshape_for_groupwise(input, zero_point, group_size, dim)

    # Quantization using STE - these functions are already differentiable
    if num_bits == 1:
        if zero_point is not None:
            # For 1-bit asymmetric: map negative to 0, positive to 1
            q_w = (input_reshaped >= zero_point_expanded).float()
        else:
            q_w = input_reshaped.sign()
        w_q = q_w * alpha_expanded
        if zero_point is not None:
            w_q = w_q + zero_point_expanded
    else:
        if zero_point is not None:
            # Asymmetric quantization with STE
            normalized = (input_reshaped - zero_point_expanded) / alpha_expanded
            q_w = clamp_ste(round_ste(normalized), Qn, Qp)
            w_q = q_w * alpha_expanded + zero_point_expanded
        else:
            # Symmetric quantization with STE
            normalized = input_reshaped / alpha_expanded
            q_w = clamp_ste(round_ste(normalized), Qn, Qp)
            w_q = q_w * alpha_expanded

    # Reshape back to original shape
    w_q = w_q.view(original_shape)

    return w_q

class QuantizeLinear(nn.Linear):
    """
    量化线性层，支持多种量化方法和分组量化
    
    采用与GPTQ相同的分组方式：按输入特征维度进行分组，
    每个分组包含连续的 group_size 个输入特征。
    支持原始的逐行量化和新的分组量化两种模式。
    """
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        group_size=128,  # 分组大小参数，与GPTQ保持一致
        enable_groupwise=False  # 是否启用分组量化
    ):
        """
        初始化量化线性层
        
        :param kargs: nn.Linear的参数（in_features, out_features等）
        :param symmetric: 是否使用对称量化
        :param bias: 是否使用偏置（通常设为False）
        :param w_bits: 权重量化位数
        :param weight_layerwise: 是否使用层级量化（用于原始量化方法）
        :param group_size: 分组大小，每组包含多少个输入特征
        :param enable_groupwise: 是否启用分组量化
        """
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.group_size = group_size
        self.enable_groupwise = enable_groupwise
        self.sym = symmetric
        
        # 初始化权重量化参数
        if self.w_bits < 16:
            if self.enable_groupwise:
                # === 分组量化：采用与GPTQ相同的分组方式 ===
                out_features, in_features = self.weight.shape
                # 计算分组数量（向上取整）
                num_groups = (in_features + group_size - 1) // group_size
                
                # 创建分组索引，与GPTQ保持一致
                self.register_buffer('g_idx', 
                    torch.tensor([i // group_size for i in range(in_features)], 
                               dtype=torch.int32))
                
                # 每个输出特征的每个分组都有独立的缩放因子
                # 形状：[out_features, num_groups]
                self.scale = nn.Parameter(torch.Tensor(out_features, num_groups))
                if not self.sym:
                    self.zero = nn.Parameter(torch.Tensor(out_features, num_groups))
                # 存储分组信息用于调试和兼容性
                self.num_groups = num_groups
                self.padded_in_features = num_groups * group_size
            else:
                # === 原始的逐行量化 ===
                # 每个输出特征一个缩放因子
                # 形状：[out_features, 1]
                self.scale = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
    
    def forward(self, input_):
        """
        前向传播
        
        :param input_: 输入张量
        :return: 量化后的线性变换结果
        """
        # 确保权重是2D张量
        assert len(self.weight.size()) == 2
        real_weights = self.weight
        
        if self.w_bits >= 16:
            # 不量化，直接使用原始权重
            weight = self.weight
        elif self.enable_groupwise:
            # 使用分组LSQ量化（带STE）- 直接调用可微分函数
            weight = lsq_groupwise_quantize(
                real_weights,
                self.scale,
                self.w_bits,
                self.group_size,
                -1,
                self.zero if not self.sym else None,
            ).to(input_.dtype)
    
        # 执行线性变换
        out = nn.functional.linear(input_, weight)
        
        # 添加偏置（如果有的话）
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
            
        return out
       