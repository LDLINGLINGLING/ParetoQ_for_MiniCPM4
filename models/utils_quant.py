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
    def forward(ctx, input, alpha, num_bits, group_size):
        """
        前向传播：执行分组LSQ量化
        
        :param ctx: PyTorch自动求导上下文，用于保存反向传播需要的信息
        :param input: 输入权重张量，形状为 [out_features, in_features]
                     通常是神经网络层的权重矩阵
        :param alpha: 每组的缩放因子，形状为 [out_features, num_groups]
                     每个输出特征的每个分组都有独立的缩放因子
        :param num_bits: 量化位数，支持1-4位量化
                        1位对应二值化，2-4位对应多级量化
        :param group_size: 每组的大小，即每组包含多少个输入特征
                          input_features会被分成若干个这样大小的组
        :return: 分组量化后的权重，形状与input相同 [out_features, in_features]
        """
        # 保存量化参数到上下文，供反向传播使用
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
        
        # 计算分组相关的维度信息
        out_features, in_features = input.shape
        # 计算需要多少个组（向上取整）
        num_groups = (in_features + group_size - 1) // group_size
        
        # === 采用与GPTQ相同的分组方式 ===
        # 创建分组索引，与GPTQ保持一致
        g_idx = torch.tensor([i // group_size for i in range(in_features)], 
                           dtype=torch.int32, device=input.device)
        
        # 计算梯度缩放因子，用于反向传播时的梯度归一化
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
            
            # === 执行量化操作 ===
            if num_bits == 1:
                # 1位量化：直接取符号，结果为 -1 或 +1
                q_w_group = w_group.sign()
            else:
                # 多位量化：先归一化，然后四舍五入，最后限制在量化范围内
                q_w_group = (w_group / alpha_group).round().clamp(Qn, Qp)
                
            # === 反量化：将量化后的值乘以缩放因子得到最终结果 ===
            w_q_group = q_w_group * alpha_group
            
            # 将量化后的权重放回对应位置
            w_q[:, start_col:end_col] = w_q_group
        
        # === 保存前向传播信息供反向传播使用 ===
        ctx.save_for_backward(input, alpha, g_idx)
        ctx.other = grad_scale, Qn, Qp, out_features, in_features
        
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
        :return: (grad_input, grad_alpha, None, None)
                grad_input: 对输入权重的梯度
                grad_alpha: 对缩放因子的梯度
                后两个None对应num_bits和group_size（不需要梯度）
        """
        # 如果不量化，直接传递梯度
        if ctx.num_bits >= 16:
            return grad_output, None, None, None
            
        # === 从上下文恢复保存的信息 ===
        input, alpha, g_idx = ctx.saved_tensors
        grad_scale, Qn, Qp, out_features, in_features = ctx.other
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
            # 小于量化下界的权重位置
            indicate_small = (q_w < Qn).float()
            # 大于量化上界的权重位置
            indicate_big = (q_w > Qp).float()
            # 在量化范围内的权重位置（这些位置会传递梯度）
            indicate_middle = 1.0 - indicate_small - indicate_big
            
            # === 计算缩放因子alpha的梯度 ===
            if ctx.num_bits == 1:
                # 1位量化：alpha的梯度与权重符号相关
                grad_alpha_group = (w_group.sign() * grad_output_group * grad_scale).sum(dim=-1, keepdim=True)
            else:
                # 多位量化：alpha的梯度包含三部分
                grad_alpha_group = ((indicate_small * Qn + indicate_big * Qp + 
                                   indicate_middle * (-q_w + q_w.round())) * 
                                  grad_output_group * grad_scale).sum(dim=-1, keepdim=True)
            
            # === 计算输入权重的梯度 ===
            # 使用直通估计器：只有在量化范围内的权重才传递梯度
            grad_input_group = indicate_middle * grad_output_group
            
            # 将梯度放回对应位置
            grad_input[:, start_col:end_col] = grad_input_group
            grad_alpha[:, group_id:group_id+1] = grad_alpha_group
        
        # 返回梯度：(输入梯度, alpha梯度, num_bits梯度=None, group_size梯度=None)
        return grad_input, grad_alpha, None, None


class GroupedStretchedElasticQuant(torch.autograd.Function):
    """
    分组Stretched Elastic量化
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, group_size):

        ctx.num_bits = num_bits
        ctx.group_size = group_size
        
        if num_bits >= 16:
            return input
            
        eps = torch.tensor(1e-5, device=alpha.device, dtype=alpha.dtype)
        alpha = torch.where(alpha > eps, alpha, eps)
        
        out_features, in_features = input.shape
        num_groups = (in_features + group_size - 1) // group_size
        
        # 将输入重塑为分组形式
        padded_in_features = num_groups * group_size
        if in_features != padded_in_features:
            padding = torch.zeros(out_features, padded_in_features - in_features,
                                device=input.device, dtype=input.dtype)
            input_padded = torch.cat([input, padding], dim=1)
        else:
            input_padded = input
            
        input_grouped = input_padded.view(out_features, num_groups, group_size)
        alpha_expanded = alpha.unsqueeze(-1)
        
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
        
        ctx.save_for_backward(input_grouped, alpha)
        ctx.other = grad_scale, Qn, Qp, out_features, in_features, n_levels, shift, clip_val
        
        if num_bits == 1:
            q_w = input_grouped.sign()
        else:
            q_w = (torch.round(torch.clamp(input_grouped / alpha_expanded, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
            
        w_q_grouped = q_w * alpha_expanded
        w_q_padded = w_q_grouped.view(out_features, padded_in_features)
        w_q = w_q_padded[:, :in_features]
        
        return w_q
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None
            
        input_grouped, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, out_features, in_features, n_levels, shift, clip_val = ctx.other
        
        num_groups = input_grouped.shape[1]
        group_size = ctx.group_size
        padded_in_features = num_groups * group_size
        
        if in_features != padded_in_features:
            padding = torch.zeros(out_features, padded_in_features - in_features,
                                device=grad_output.device, dtype=grad_output.dtype)
            grad_output_padded = torch.cat([grad_output, padding], dim=1)
        else:
            grad_output_padded = grad_output
            
        grad_output_grouped = grad_output_padded.view(out_features, num_groups, group_size)
        alpha_expanded = alpha.unsqueeze(-1)
        
        q_w = input_grouped / alpha_expanded
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        
        if ctx.num_bits == 1:
            grad_alpha = (input_grouped.sign() * grad_output_grouped * grad_scale).sum(dim=-1)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + 
                          indicate_middle * (-q_w + (torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels)) * 
                         grad_output_grouped * grad_scale).sum(dim=-1)
        
        grad_input_grouped = indicate_middle * grad_output_grouped
        grad_input_padded = grad_input_grouped.view(out_features, padded_in_features)
        grad_input = grad_input_padded[:, :in_features]
        
        return grad_input, grad_alpha, None, None


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        group_size=128,  # 新增：分组大小参数
        enable_groupwise=False,  # 新增：是否启用分组量化
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.group_size = group_size
        self.enable_groupwise = enable_groupwise
        
        # params for weight quant
        if self.w_bits < 16:
            if self.enable_groupwise:
                # 分组量化：每组一个缩放因子
                out_features, in_features = self.weight.shape
                num_groups = (in_features + group_size - 1) // group_size
                self.weight_clip_val = nn.Parameter(torch.Tensor(out_features, num_groups))
            else:
                # 原始的逐行量化
                self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight
        elif self.enable_groupwise:
            # 使用分组量化
            if self.w_bits == 2 or self.w_bits == 0:
                weight = GroupedStretchedElasticQuant.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.group_size,
                ).to(input_.dtype)
            elif self.w_bits <= 4:
                weight = GroupedLsqQuantization.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.group_size,
                ).to(input_.dtype)
            else:
                raise NotImplementedError
        else:
            # 使用原始的逐行量化
            if self.w_bits == 2 or self.w_bits == 0:
                weight = StretchedElasticQuant.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)
            elif self.w_bits <= 4:
                weight = LsqBinaryTernaryExtension.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)
            else:
                raise NotImplementedError

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
