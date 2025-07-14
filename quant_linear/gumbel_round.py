"""
修复后的LSQ GroupWise Gumbel扩展 - 可微分量化实现
保持Gumbel梯度计算方式，但采用与utils_group_zero_min.py一致的代码结构
"""

import torch
import torch.nn.functional as F
import math
import torch.nn as nn

def gumbel_round(x, temperature=1.0, hard=True):
    """
    基于Gumbel-Softmax近似的可微分舍入函数
    
    参数：
        x: 输入张量
        temperature: 温度参数，越小越接近离散
        hard: 是否使用硬采样（前向离散，反向连续）
    
    返回：
        近似舍入后的张量
    """
    # 将连续值转换为离散选择问题：floor vs ceil
    logits = torch.stack([
        -torch.abs(x - torch.floor(x)),  # 选择floor的负距离（越近概率越大）
        -torch.abs(x - torch.ceil(x))    # 选择ceil的负距离（越近概率越大）
    ], dim=-1)
    
    # 添加Gumbel噪声进行采样
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        # 硬采样：前向传播使用离散值，反向传播使用连续梯度
        y_hard = torch.zeros_like(y_soft)
        y_hard[..., 0] = (y_soft[..., 0] > y_soft[..., 1]).float()
        y_hard[..., 1] = 1 - y_hard[..., 0]
        y = y_hard - y_soft.detach() + y_soft  # 直通估计器技巧
        return torch.floor(x) * y[..., 0] + torch.ceil(x) * y[..., 1]
    else:
        # 软采样：返回期望值
        return torch.floor(x) * y_soft[..., 0] + torch.ceil(x) * y_soft[..., 1]

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

class LsqGroupWiseExtension(torch.autograd.Function):
    """
    Group-wise quantization extension based on Learned Step-size Quantization.
    Modified from the original LSQ implementation to support group-wise quantization
    with optional asymmetric quantization (zero point).
    Uses Gumbel-Softmax for improved gradient flow in alpha calculation.
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, group_size, dim=-1, zero_point=None, temperature=0.1):
        """
        :param input: input to be quantized
        :param alpha: the step size (per group)
        :param num_bits: quantization bits
        :param group_size: size of each group for quantization
        :param dim: dimension along which to group (default: -1, last dimension)
        :param zero_point: zero point for asymmetric quantization (per group), None for symmetric
        :return: quantized output
        """
        ctx.num_bits = num_bits
        ctx.group_size = group_size
        ctx.dim = dim
        ctx.asymmetric = zero_point is not None

        if num_bits >= 16:
            return input

        # Calculate quantization range
        if ctx.asymmetric:
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
        eps = torch.tensor(1e-5, device=alpha.device, dtype=alpha.dtype)  # Changed from 1e-4 to 1e-5 per math doc
        alpha = torch.clamp(alpha, min=eps)
        
        # Check for NaN/Inf in alpha
        if torch.any(torch.isnan(alpha)) or torch.any(torch.isinf(alpha)):
            print("Warning: NaN or Inf detected in alpha during forward pass")
            alpha = torch.where(torch.isnan(alpha) | torch.isinf(alpha), eps, alpha)

        # Calculate gradient scale - fixed per math formulas
        input_numel = input.numel()
        if Qp == 0:
            grad_scale = 1.0 / math.sqrt(max(input_numel, 1))
        else:
            grad_scale = 1.0 / math.sqrt(max(input_numel * abs(Qp), 1))

        # Reshape input and alpha for group-wise processing
        original_shape = input.shape
        input_reshaped, alpha_expanded = _reshape_for_groupwise(input, alpha, group_size, dim)

        # Handle zero point for asymmetric quantization
        zero_point_expanded = None
        if ctx.asymmetric:
            # 计算每个组的最小值作为零点
            # input_reshaped shape: [in_features, num_groups, group_size]
            # 沿着最后一个维度（group_size）找最小值
            zero_point_min = input_reshaped.min(dim=-1, keepdim=True)[0]  # shape: [in_features, num_groups, 1]
            zero_point_expanded = zero_point_min.expand_as(input_reshaped)  # shape: [in_features, num_groups, group_size]
            
            # 如果用户提供了zero_point参数，我们仍然保存它用于backward，但实际使用计算的最小值
            if zero_point is not None:
                zero_point_reshaped, _ = _reshape_for_groupwise(input, zero_point, group_size, dim)
            else:
                # 如果用户没有提供zero_point，我们创建一个与alpha相同形状的零张量作为占位符
                zero_point = torch.zeros_like(alpha)
                zero_point_reshaped, _ = _reshape_for_groupwise(input, zero_point, group_size, dim)
            
            zero_point_expanded = zero_point_expanded.detach()  # Zero point doesn't need gradients in this implementation

        ctx.save_for_backward(input, alpha, zero_point)
        ctx.other = grad_scale, Qn, Qp, original_shape

        # Quantization
        if num_bits == 1:
            if ctx.asymmetric:
                # For 1-bit asymmetric: map negative to 0, positive to 1
                q_w = (input_reshaped >= zero_point_expanded).float()
            else:
                q_w = input_reshaped.sign()
        else:
            if ctx.asymmetric:
                # Asymmetric quantization: q = round((x - zero_point) / alpha) + zero_point
                q_w = ((input_reshaped - zero_point_expanded) / alpha_expanded).round().clamp(Qn, Qp)
                w_q = q_w * alpha_expanded + zero_point_expanded
            else:
                # Symmetric quantization: q = round(x / alpha)
                q_w = (input_reshaped / alpha_expanded).round().clamp(Qn, Qp)
                w_q = q_w * alpha_expanded

        if num_bits == 1:
            w_q = q_w * alpha_expanded
            if ctx.asymmetric:
                w_q = w_q + zero_point_expanded

        # Reshape back to original shape
        w_q = w_q.view(original_shape)

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None, None, None, None

        input_, alpha, zero_point = ctx.saved_tensors
        grad_scale, Qn, Qp, original_shape = ctx.other
        group_size = ctx.group_size
        dim = ctx.dim
        temperature = ctx.temperature
        
        # Check for NaN/Inf in grad_output
        if torch.any(torch.isnan(grad_output)) or torch.any(torch.isinf(grad_output)):
            print("Warning: NaN or Inf detected in grad_output")
            grad_output = torch.where(torch.isnan(grad_output) | torch.isinf(grad_output), 
                                    torch.zeros_like(grad_output), grad_output)

        # Reshape for group-wise processing
        input_reshaped, alpha_expanded = _reshape_for_groupwise(input_, alpha, group_size, dim)
        grad_output_reshaped = grad_output.view(input_reshaped.shape)

        # Handle zero point
        zero_point_expanded = None
        if ctx.asymmetric:
            # 重新计算每个组的最小值作为零点（与forward保持一致）
            # input_reshaped shape: [out_features, num_groups, group_size]
            zero_point_min = input_reshaped.min(dim=-1, keepdim=True)[0]  # shape: [out_features, num_groups, 1]
            zero_point_expanded = zero_point_min.expand_as(input_reshaped)  # shape: [out_features, num_groups, group_size]
            zero_point_expanded = zero_point_expanded.detach()  # 不需要梯度

        if ctx.asymmetric:
            q_w = (input_reshaped - zero_point_expanded) / alpha_expanded
        else:
            q_w = input_reshaped / alpha_expanded

        # Calculate indicators
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        # Calculate gradient w.r.t. alpha using Gumbel-based approach for better gradient flow
        if ctx.num_bits == 1:
            if ctx.asymmetric:
                grad_alpha_term = ((input_reshaped >= zero_point_expanded).float()) * grad_output_reshaped * grad_scale
            else:
                grad_alpha_term = input_reshaped.sign() * grad_output_reshaped * grad_scale
        else:
            # Use Gumbel-Softmax for smooth gradient calculation
            q_w_soft = gumbel_round(q_w, temperature=temperature, hard=False)
            
            if ctx.asymmetric:
                # Gumbel-based gradient for asymmetric quantization
                grad_alpha_term = (
                    indicate_small * Qn  
                    + indicate_big * Qp
                    + indicate_middle * (-q_w_soft)  # Gumbel provides smooth gradients
                ) * grad_output_reshaped * grad_scale
            else:
                # Gumbel-based gradient for symmetric quantization
                grad_alpha_term = (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_w_soft)  # Gumbel provides smooth gradients
                ) * grad_output_reshaped * grad_scale

        # Clamp gradient term to prevent overflow
        grad_alpha_term = torch.clamp(grad_alpha_term, -1e6, 1e6)

        # Sum over group dimension to get gradient for each group
        grad_alpha = _sum_over_groups(grad_alpha_term, group_size, dim, alpha.shape)
        
        # Additional gradient clipping for alpha
        grad_alpha = torch.clamp(grad_alpha, -1e3, 1e3)

        # Calculate gradient w.r.t. zero_point - keep original logic
        # 注意：由于我们将zero_point设置为每组的最小值，它是输入数据的函数
        # 在这种情况下，我们不需要计算zero_point的梯度，因为它会自动随着输入数据更新
        grad_zero_point = None
        if ctx.asymmetric:
            # 保持原有的梯度计算逻辑，但由于zero_point是数据驱动的，实际上不会更新
            if ctx.num_bits == 1:
                # For 1-bit: w_q = I(x >= z) * alpha + z
                # ∂w_q/∂z = -δ(x - z) * alpha + 1 ≈ 1 (approximation for STE)
                grad_zero_point_term = grad_output_reshaped * grad_scale
            else:
                # Complete zero point gradient with both contributions
                # Main contribution: ∂z/∂z = 1 (zero point directly adds to output)
                main_contribution = grad_output_reshaped * grad_scale
                
                # Additional contribution from quantization in middle region
                # In middle region: ∂(round((x-z)/α) * α)/∂z ≈ -1 (STE approximation)
                quantization_contribution = -indicate_middle * grad_output_reshaped * grad_scale
                
                grad_zero_point_term = main_contribution + quantization_contribution

            # Clamp zero point gradient
            grad_zero_point_term = torch.clamp(grad_zero_point_term, -1e6, 1e6)
            grad_zero_point = _sum_over_groups(grad_zero_point_term, group_size, dim, zero_point.shape)
            grad_zero_point = torch.clamp(grad_zero_point, -1e3, 1e3)
            
            # 由于zero_point是数据驱动的（每组最小值），我们将其梯度设置为None
            grad_zero_point = None

        # Calculate gradient w.r.t. input - standard approach
        if ctx.asymmetric:
            grad_input = (
                indicate_small * 1
                + indicate_middle * 1.0
                + indicate_big * 1
            ) * grad_output_reshaped
        else:
            grad_input = (
                indicate_small * 1
                + indicate_middle * 1.0
                + indicate_big * 1
            ) * grad_output_reshaped
        
        grad_input = grad_input.view(original_shape)
        
        # Final NaN check
        if torch.any(torch.isnan(grad_alpha)):
            print("Warning: NaN detected in grad_alpha, setting to zero")
            grad_alpha = torch.zeros_like(grad_alpha)
        
        if grad_zero_point is not None and torch.any(torch.isnan(grad_zero_point)):
            print("Warning: NaN detected in grad_zero_point, setting to zero")
            grad_zero_point = torch.zeros_like(grad_zero_point)

        return grad_input, grad_alpha, None, None, None, grad_zero_point, None

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
        enable_groupwise=False,  # 是否启用分组量化
        temperature=0.1,  # Gumbel温度参数
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
        :param temperature: Gumbel-Softmax温度参数
        """
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.group_size = group_size
        self.enable_groupwise = enable_groupwise
        self.sym = symmetric
        self.temperature = temperature
        
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
            # 使用分组LSQ量化（带Gumbel-Softmax）
            weight = LsqGroupWiseExtension.apply(
                real_weights,
                self.scale,
                self.w_bits,
                self.group_size,
                -1,
                self.zero if not self.sym else None,
                self.temperature,
            ).to(input_.dtype)
    
        # 执行线性变换
        out = nn.functional.linear(input_, weight)
        
        # 添加偏置（如果有的话）
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
            
        return out