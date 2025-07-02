# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn

class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(
                    torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                        - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                    - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class GroupedLsqQuantization(torch.autograd.Function):
    """
    分组LSQ量化，支持将权重按组进行量化，每组有独立的缩放因子
    
    LSQ (Learned Step-size Quantization) 的分组版本，将权重矩阵按列分组，
    每组使用独立的量化缩放因子。这样可以更好地适应权重分布的差异，
    提高量化精度。
    
    采用与GPTQ相同的分组方式：按输入特征维度进行分组，
    形状为 [out_features, in_features] 的权重矩阵中，
    每个分组包含连续的 group_size 个输入特征。
    """
    @staticmethod
    def forward(ctx, input, alpha, zero, num_bits, group_size):
        """
        前向传播：执行分组LSQ量化
        
        :param ctx: PyTorch自动求导上下文，用于保存反向传播需要的信息
        :param input: 输入权重张量，形状为 [out_features, in_features]
        :param alpha: 每组的缩放因子，形状为 [out_features, num_groups]
        :param zero: 每组的零点，形状为 [out_features, num_groups]
        :param num_bits: 量化位数，支持1-4位量化
        :param group_size: 每组的大小，即每组包含多少个输入特征
        :return: 分组量化后的权重，形状与input相同 [out_features, in_features]
        """
        # 保存量化参数到上下文，供反向传播使用
        if torch.isnan(zero).any():
            raise ValueError("zero weights contain NaN values. Please check the input and parameters.")

        if torch.isnan(alpha).any():
            raise ValueError("alpha weights contain NaN values. Please check the input and parameters.")
        ctx.num_bits = num_bits
        ctx.group_size = group_size
        
        # 如果量化位数>=16，相当于不量化，直接返回原始输入
        if num_bits >= 16:
            return input
            
        # 根据量化位数确定量化范围
        if num_bits == 1 or num_bits == 0:
            # 1位量化：二值化，范围为 [-1, 1]
            Qn = -1  # 量化下界
            Qp = 1   # 量化上界
        else:
            # 多位量化：对称量化，范围为 [-(2^(n-1)), 2^(n-1)-1]
            # 例如3位量化范围为 [-4, 3]
            Qn = -(2 ** (num_bits - 1))      # 量化下界
            Qp = 2 ** (num_bits - 1) - 1     # 量化上界
            
        # 设置数值稳定性的最小值，防除零错误
        eps = torch.tensor(1e-5, device=alpha.device, dtype=alpha.dtype)
        # 确保alpha不会太小，避免数值不稳定
        alpha = torch.where(alpha > eps, alpha, eps)
        alpha = torch.where(alpha < -eps, alpha, -eps)
        
        # 计算分组相关的维度信息
        out_features, in_features = input.shape
        # 计算需要多少个组（向上取整）
        num_groups = (in_features + group_size - 1) // group_size
        
        # === 在forward内初始化g_idx，确保与当前设备和输入匹配 ===
        g_idx = torch.tensor([i // group_size for i in range(in_features)], 
                           dtype=torch.int32, device=input.device)
        
        # === 修正grad_scale计算 ===
        # 梯度缩放因子应该基于每组的实际元素数量，而不是整个张量
        # 使用sqrt(group_size * out_features)来归一化每组的梯度
        if Qp != 0:
            grad_scale = 1.0 / math.sqrt(group_size * out_features * Qp)
        else:
            grad_scale = 1.0 / math.sqrt(group_size * out_features)
        
        # === 使用向量化操作减少for循环 ===
        # 将输入重塑为分组形状，便于批量处理
        padded_in_features = num_groups * group_size
        if in_features < padded_in_features:
            # 如果需要，对输入进行填充
            input_padded = torch.cat([input, torch.zeros(out_features, padded_in_features - in_features, 
                                                       device=input.device, dtype=input.dtype)], dim=1)
        else:
            input_padded = input
        
        # 重塑为 [out_features, num_groups, group_size]
        w_grouped = input_padded.view(out_features, num_groups, group_size)
        
        # 扩展alpha和zero的维度以匹配分组权重
        alpha_expanded = alpha.unsqueeze(-1)  # [out_features, num_groups, 1]
        
        # === 实现zero_scale逻辑 ===
        if zero is not None:
            zero_expanded = zero.unsqueeze(-1)  # [out_features, num_groups, 1]
            # 对zero进行缩放，确保其在合理范围内
            zero_scale = 0.1  # 零点缩放因子，可以作为超参数调整
            zero_scaled = zero_expanded# * zero_scale
        else:
            zero_scaled = torch.zeros_like(alpha_expanded)
        
        # === 执行向量化量化操作 ===
        if num_bits == 1:
            # 1位量化：直接取符号
            q_w_grouped = w_grouped.sign()
            # 对于1位量化，零点通常不使用
            w_q_grouped = q_w_grouped * alpha_expanded
        else:
            
            # 多位量化：先减去零点，归一化，然后四舍五入，最后限制在量化范围内
            normalized_w = (w_grouped - zero_scaled) / alpha_expanded
            q_w_grouped = normalized_w.round().clamp(Qn, Qp)
            if torch.isnan(zero_scaled).any().item() :
                raise ValueError("zero_scaled weights contain NaN values. Please check the input and parameters.")
            if torch.isnan(w_grouped).any().item() != torch.isnan(normalized_w).any().item():
                raise ValueError("divide weights contain NaN values. Please check the input and parameters.")

            # 反量化：恢复到原始尺度
            w_q_grouped = q_w_grouped * alpha_expanded + zero_scaled
        
        # 重塑回原始形状
        w_q_padded = w_q_grouped.view(out_features, padded_in_features)
        
        # 如果之前进行了填充，需要截取到原始大小
        w_q = w_q_padded[:, :in_features]
        
        # === 保存前向传播信息供反向传播使用 ===
        ctx.save_for_backward(input, alpha, zero, g_idx)
        ctx.other = grad_scale, Qn, Qp, out_features, in_features, zero_scale if zero is not None else 0.0
        if torch.isnan(w_q).any():
            raise ValueError("Quantized weights contain NaN values. Please check the input and parameters.")
        return w_q
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算梯度
        
        计算量化操作对输入和缩放因子的梯度。
        使用直通估计器(Straight-Through Estimator)的思想，
        只有在量化范围内的权重才传递梯度。
        
        :param ctx: 前向传播保存的上下文信息
        :param grad_output: 来自上层的梯度，形状为 [out_features, in_features]
        :return: (grad_input, grad_alpha, grad_zero, None, None)
        """

        # 如果不量化，直接传递梯度
        if ctx.num_bits >= 16:
            return grad_output, None, None, None, None
        if torch.isnan(grad_output).any():
            raise ValueError("Gradient output contains NaN values. Please check the input and parameters.")
        # === 从上下文恢复保存的信息 ===
        input, alpha, zero, g_idx = ctx.saved_tensors
        grad_scale, Qn, Qp, out_features, in_features, zero_scale = ctx.other
        group_size = ctx.group_size
        
        # 计算分组数量
        num_groups = (in_features + group_size - 1) // group_size
        
        # === 使用向量化操作计算梯度 ===
        padded_in_features = num_groups * group_size
        
        # 对梯度进行填充（如果需要）
        if in_features < padded_in_features:
            grad_output_padded = torch.cat([grad_output, torch.zeros(out_features, padded_in_features - in_features, 
                                                                   device=grad_output.device, dtype=grad_output.dtype)], dim=1)
            input_padded = torch.cat([input, torch.zeros(out_features, padded_in_features - in_features, 
                                                       device=input.device, dtype=input.dtype)], dim=1)
        else:
            grad_output_padded = grad_output
            input_padded = input
        
        # 重塑为分组形状
        grad_output_grouped = grad_output_padded.view(out_features, num_groups, group_size)
        w_grouped = input_padded.view(out_features, num_groups, group_size)
        
        # 扩展alpha维度
        alpha_expanded = alpha.unsqueeze(-1)  # [out_features, num_groups, 1]
        
        # 处理zero
        if zero is not None:
            zero_expanded = zero.unsqueeze(-1)
            zero_scaled = zero_expanded * zero_scale
        else:
            zero_expanded = torch.zeros_like(alpha_expanded)
            zero_scaled = zero_expanded
        
        # === 向量化计算量化指示器 ===
        if ctx.num_bits == 1:
            # 1位量化的梯度计算
            grad_alpha_grouped = (w_grouped.sign() * grad_output_grouped * grad_scale).sum(dim=-1, keepdim=True)
            grad_input_grouped = grad_output_grouped  # 直通估计器
            grad_zero_grouped = torch.zeros_like(alpha_expanded) if zero is not None else None
        else:
            # 计算归一化后的权重值
            q_w = (w_grouped - zero_scaled) / alpha_expanded
            # 量化指示器
            indicate_small = (q_w < Qn).float()
            indicate_big = (q_w > Qp).float()
            indicate_middle = 1.0 - indicate_small - indicate_big
            
            # 计算alpha的梯度
            grad_alpha_grouped = ((indicate_small * Qn + indicate_big * Qp + 
                                 indicate_middle * (-q_w + q_w.round())) * 
                                grad_output_grouped * grad_scale).sum(dim=-1, keepdim=True)
            
            # 计算输入权重的梯度（直通估计器）
            grad_input_grouped = indicate_middle * grad_output_grouped
            
            # 计算zero的梯度
            if zero is not None:
                grad_zero_grouped = (-(indicate_small * Qn + indicate_big * Qp + 
                                     indicate_middle * (-q_w + q_w.round())) * 
                                   grad_output_grouped * grad_scale * zero_scale).sum(dim=-1, keepdim=True)
            else:
                grad_zero_grouped = None
        
        # 重塑回原始形状
        grad_input_padded = grad_input_grouped.view(out_features, padded_in_features)
        grad_input = grad_input_padded[:, :in_features]
        
        grad_alpha = grad_alpha_grouped.squeeze(-1)  # [out_features, num_groups]
        
        if grad_zero_grouped is not None:
            grad_zero = grad_zero_grouped.squeeze(-1)  # [out_features, num_groups]
        else:
            grad_zero = None
        # import pdb
        # pdb.set_trace()
        # 返回梯度：(输入梯度, alpha梯度, zero梯度, num_bits梯度=None, group_size梯度=None)
        return grad_input, grad_alpha, grad_zero, None, None


class GroupedStretchedElasticQuant(torch.autograd.Function):
    """
    分组Stretched Elastic量化
    
    采用与GPTQ相同的分组方式：按输入特征维度进行分组，
    形状为 [out_features, in_features] 的权重矩阵中，
    每个分组包含连续的 group_size 个输入特征。
    
    Stretched Elastic量化使用特殊的量化范围和级别计算方式，
    提供更灵活的量化策略。
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, group_size):
        """
        前向传播：执行分组Stretched Elastic量化
        
        :param ctx: PyTorch自动求导上下文
        :param input: 输入权重张量，形状为 [out_features, in_features]
        :param alpha: 每组的缩放因子，形状为 [out_features, num_groups]
        :param num_bits: 量化位数
        :param group_size: 每组的大小
        :return: 分组量化后的权重，形状与input相同
        """
        ctx.num_bits = num_bits
        ctx.group_size = group_size
        
        if num_bits >= 16:
            return input
            
        eps = torch.tensor(1e-5, device=alpha.device, dtype=alpha.dtype)
        alpha = torch.where(alpha > eps, alpha, eps)
        
        out_features, in_features = input.shape
        num_groups = (in_features + group_size - 1) // group_size
        
        # === 采用与GPTQ相同的分组方式 ===
        # 创建分组索引，与GPTQ保持一致
        g_idx = torch.tensor([i // group_size for i in range(in_features)], 
                           dtype=torch.int32, device=input.device)
        
        # Stretched Elastic量化的特殊参数
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
            
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        
        grad_scale = 1.0 / math.sqrt(input.numel() * Qp) if Qp else 1.0 / math.sqrt(input.numel())
        
        # === 执行分组量化 ===
        w_q = torch.zeros_like(input)
        
        # 按组进行量化，与GPTQ的分组逻辑一致
        for group_id in range(num_groups):
            # 计算当前组的列范围
            start_col = group_id * group_size
            end_col = min(start_col + group_size, in_features)
            
            # 获取当前组的权重
            w_group = input[:, start_col:end_col]  # [out_features, current_group_size]
            
            # 获取当前组的缩放因子
            alpha_group = alpha[:, group_id:group_id+1]  # [out_features, 1]
            
            # === 执行Stretched Elastic量化操作 ===
            if num_bits == 1:
                # 1位量化：直接取符号
                q_w_group = w_group.sign()
            else:
                # Stretched Elastic量化：使用特殊的量化公式
                normalized = w_group / alpha_group
                clamped = torch.clamp(normalized, -clip_val, clip_val)
                scaled = clamped * n_levels - shift
                rounded = torch.round(scaled)
                q_w_group = (rounded + shift) / n_levels
                
            # === 反量化：将量化后的值乘以缩放因子得到最终结果 ===
            w_q_group = q_w_group * alpha_group
            
            # 将量化后的权重放回对应位置
            w_q[:, start_col:end_col] = w_q_group
        
        # === 保存前向传播信息供反向传播使用 ===
        ctx.save_for_backward(input, alpha, g_idx)
        ctx.other = grad_scale, Qn, Qp, out_features, in_features, n_levels, shift, clip_val
        
        return w_q
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算梯度
        
        :param ctx: 前向传播保存的上下文信息
        :param grad_output: 来自上层的梯度，形状为 [out_features, in_features]
        :return: (grad_input, grad_alpha, None, None)
        """
        if ctx.num_bits >= 16:
            return grad_output, None, None, None
            
        # === 从上下文恢复保存的信息 ===
        input, alpha, g_idx = ctx.saved_tensors
        grad_scale, Qn, Qp, out_features, in_features, n_levels, shift, clip_val = ctx.other
        group_size = ctx.group_size
        
        # 计算分组数量
        num_groups = (in_features + group_size - 1) // group_size
        
        # 初始化梯度
        grad_input = torch.zeros_like(input)
        grad_alpha = torch.zeros_like(alpha)
        
        # === 按组计算梯度，与前向传播保持一致 ===
        for group_id in range(num_groups):
            # 计算当前组的列范围
            start_col = group_id * group_size
            end_col = min(start_col + group_size, in_features)
            
            # 获取当前组的权重和梯度
            w_group = input[:, start_col:end_col]  # [out_features, current_group_size]
            grad_output_group = grad_output[:, start_col:end_col]  # [out_features, current_group_size]
            
            # 获取当前组的缩放因子
            alpha_group = alpha[:, group_id:group_id+1]  # [out_features, 1]
            
            # === 计算量化指示器 ===
            # 计算归一化后的权重值
            q_w = w_group / alpha_group
            # 使用clip_val作为边界，而不是原来的Qn/Qp
            indicate_small = (q_w < -clip_val).float()
            indicate_big = (q_w > clip_val).float()
            indicate_middle = 1.0 - indicate_small - indicate_big
            
            # === 计算缩放因子alpha的梯度 ===
            if ctx.num_bits == 1:
                # 1位量化：alpha的梯度与权重符号相关
                grad_alpha_group = (w_group.sign() * grad_output_group * grad_scale).sum(dim=-1, keepdim=True)
            else:
                # Stretched Elastic量化的梯度计算
                # 重新计算量化值用于梯度计算
                clamped_q_w = torch.clamp(q_w, -clip_val, clip_val)
                scaled = clamped_q_w * n_levels - shift
                rounded = torch.round(scaled)
                quantized_normalized = (rounded + shift) / n_levels
                
                grad_alpha_group = ((indicate_small * Qn + indicate_big * Qp + 
                                   indicate_middle * (-q_w + quantized_normalized)) * 
                                  grad_output_group * grad_scale).sum(dim=-1, keepdim=True)
            
            # === 计算输入权重的梯度 ===
            # 使用直通估计器：只有在量化范围内的权重才传递梯度
            grad_input_group = indicate_middle * grad_output_group
            
            # 将梯度放回对应位置
            grad_input[:, start_col:end_col] = grad_input_group
            grad_alpha[:, group_id:group_id+1] = grad_alpha_group
        
        # 返回梯度：(输入梯度, alpha梯度, num_bits梯度=None, group_size梯度=None)
        return grad_input, grad_alpha, None, None


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
        
        # 初始化权重量化参数
        if self.w_bits < 16:
            if self.enable_groupwise:
                # === 分组量化：采用与GPTQ相同的分组方式 ===
                out_features, in_features = self.weight.shape
                # 计算分组数量（向上取整）
                num_groups = (in_features + group_size - 1) // group_size
                
                # === g_idx应该在这里初始化并注册为buffer ===
                # 这样可以确保它随模型一起保存和加载，并自动处理设备转移
                self.register_buffer('g_idx', 
                    torch.tensor([i // group_size for i in range(in_features)], 
                               dtype=torch.int32))
                
                # 每个输出特征的每个分组都有独立的缩放因子
                # 形状：[out_features, num_groups]
                scale_tensor = torch.full((out_features, num_groups), 0.01, dtype=torch.float32)
                zero_tensor = torch.zeros(out_features, num_groups, dtype=torch.float32)
                
                # 确保创建的tensor是有效的
                assert torch.all(torch.isfinite(scale_tensor)), "Scale tensor contains invalid values"
                assert torch.all(torch.isfinite(zero_tensor)), "Zero tensor contains invalid values"
                
                self.scale = nn.Parameter(scale_tensor)
                self.zero = nn.Parameter(zero_tensor)
            else:
                # === 原始的逐行量化 ===
                # 每个输出特征一个缩放因子
                # 形状：[out_features, 1]
                self.scale = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                nn.init.constant_(self.scale, 0.01)
    
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
            # === 使用分组量化（与GPTQ分组方式一致）===
            if self.w_bits == 2 or self.w_bits == 0:
                # 使用分组Stretched Elastic量化
                weight = GroupedStretchedElasticQuant.apply(
                    real_weights,
                    self.scale,
                    self.w_bits,
                    self.group_size,
                ).to(input_.dtype)
            elif self.w_bits <= 4:
                # 使用分组LSQ量化
                weight = GroupedLsqQuantization.apply(
                    real_weights,
                    self.scale,
                    self.zero,
                    self.w_bits,
                    self.group_size,
                ).to(input_.dtype)
            else:
                raise NotImplementedError(f"分组量化不支持 {self.w_bits} 位量化")
        else:
            # === 使用原始的逐行量化 ===
            if self.w_bits == 2 or self.w_bits == 0:
                # 使用原始Stretched Elastic量化
                weight = StretchedElasticQuant.apply(
                    real_weights,
                    self.scale,
                    self.w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)
            elif self.w_bits <= 4:
                # 使用原始LSQ量化
                weight = LsqBinaryTernaryExtension.apply(
                    real_weights,
                    self.scale,
                    self.w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)
            else:
                raise NotImplementedError(f"逐行量化不支持 {self.w_bits} 位量化")
        if torch.isnan(weight).any().item():
            raise ValueError("weight contains NaN values. Please check the input and parameters.")
        if torch.isnan(input_).any().item():
            raise ValueError("input_ contains NaN values. Please check the input and parameters.")
        # 执行线性变换
        out = nn.functional.linear(input_, weight)
        
        # 添加偏置（如果有的话）
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        if torch.isnan(out).any().item():
            raise ValueError("Output contains NaN values. Please check the input and parameters.")
        return out
    
    def extra_repr(self):
        """
        返回层的额外描述信息
        """
        info = f'in_features={self.in_features}, out_features={self.out_features}'
        info += f', w_bits={self.w_bits}'
        
        if self.enable_groupwise:
            info += f', group_size={self.group_size}, num_groups={self.num_groups}'
            info += ', groupwise=True'
        else:
            info += ', groupwise=False'
            
        if self.weight_layerwise:
            info += ', weight_layerwise=True'
            
        return info
    
    def get_grouping_info(self):
        """
        获取分组信息，用于调试和与GPTQ框架的兼容性
        
        :return: 包含分组信息的字典
        """
        if not self.enable_groupwise:
            return None
            
        return {
            'group_size': self.group_size,
            'num_groups': self.num_groups,
            'g_idx': self.g_idx,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'padded_in_features': self.padded_in_features,
            'scale_shape': self.scale.shape
        }
    
    def set_grouping_from_gptq(self, scale, zero, g_idx):
        """
        从GPTQ的量化参数设置分组量化参数
        用于与GPTQ框架的兼容性
        
        :param scale: GPTQ的缩放因子
        :param zero: GPTQ的零点（在LSQ中不使用，但保持接口兼容）
        :param g_idx: GPTQ的分组索引
        """
        if not self.enable_groupwise:
            raise ValueError("只有启用分组量化时才能从GPTQ设置参数")
            
        # 验证分组索引的兼容性
        expected_g_idx = torch.tensor([i // self.group_size for i in range(self.in_features)], 
                                    dtype=torch.int32, device=g_idx.device)
        
        if not torch.equal(g_idx, expected_g_idx):
            print("警告：GPTQ的分组索引与当前设置不完全匹配，可能影响兼容性")
        
        # 将GPTQ的scale转换为ParetoQ的scale格式
        if scale.shape == self.scale.shape:
            with torch.no_grad():
                self.scale.copy_(scale)
        else:
            raise ValueError(f"GPTQ的scale形状 {scale.shape} 与期望形状 {self.scale.shape} 不匹配")
