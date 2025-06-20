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
    
    与传统的逐行量化不同，分组量化在输入特征维度上进行分组，
    形状为 [out_features, in_features] 的权重矩阵被分为
    [out_features, num_groups, group_size] 的形式进行量化。
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
        
        # === 处理输入特征维度不能被group_size整除的情况 ===
        # 计算填充后的输入特征数
        padded_in_features = num_groups * group_size
        if in_features != padded_in_features:
            # 如果输入特征数不是group_size的整数倍，需要用0填充
            # 创建填充张量，形状为 [out_features, 填充长度]
            padding = torch.zeros(out_features, padded_in_features - in_features, 
                                device=input.device, dtype=input.dtype)
            # 在最后一个维度上拼接，得到填充后的输入
            input_padded = torch.cat([input, padding], dim=1)
        else:
            # 如果正好整除，不需要填充
            input_padded = input
            
        # === 重塑为分组形式 ===
        # 将 [out_features, padded_in_features] 重塑为 [out_features, num_groups, group_size]
        # 这样每个组的权重就连续存储了
        input_grouped = input_padded.view(out_features, num_groups, group_size)
        
        # === 扩展缩放因子维度 ===
        # alpha形状: [out_features, num_groups] -> [out_features, num_groups, 1]
        # 这样可以与input_grouped进行广播运算
        alpha_expanded = alpha.unsqueeze(-1)
        
        # 计算梯度缩放因子，用于反向传播时的梯度归一化
        # 如果Qp非零，按元素数和量化上界进行归一化；否则只按元素数归一化
        grad_scale = 1.0 / math.sqrt(input.numel() * Qp) if Qp else 1.0 / math.sqrt(input.numel())
        
        # === 保存前向传播信息供反向传播使用 ===
        ctx.save_for_backward(input_grouped, alpha)
        ctx.other = grad_scale, Qn, Qp, out_features, in_features
        
        # === 执行量化操作 ===
        if num_bits == 1:
            # 1位量化：直接取符号，结果为 -1 或 +1
            q_w = input_grouped.sign()
        else:
            # 多位量化：先归一化，然后四舍五入，最后限制在量化范围内
            # 1. input_grouped / alpha_expanded: 归一化到量化范围
            # 2. round(): 四舍五入到最近的整数
            # 3. clamp(Qn, Qp): 限制在量化范围内
            q_w = (input_grouped / alpha_expanded).round().clamp(Qn, Qp)
            
        # === 反量化：将量化后的值乘以缩放因子得到最终结果 ===
        w_q_grouped = q_w * alpha_expanded
        
        # === 重塑回原始形状 ===
        # 将 [out_features, num_groups, group_size] 重塑回 [out_features, padded_in_features]
        w_q_padded = w_q_grouped.view(out_features, padded_in_features)
        # 去掉之前添加的填充，恢复到原始输入特征数
        w_q = w_q_padded[:, :in_features]
        
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
        input_grouped, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, out_features, in_features = ctx.other
        
        # === 处理grad_output的形状，使其与前向传播时的分组一致 ===
        num_groups = input_grouped.shape[1]
        group_size = ctx.group_size
        padded_in_features = num_groups * group_size
        
        # 如果原始输入特征数不是group_size的整数倍，需要填充grad_output
        if in_features != padded_in_features:
            # 创建与前向传播时相同的填充
            padding = torch.zeros(out_features, padded_in_features - in_features,
                                device=grad_output.device, dtype=grad_output.dtype)
            grad_output_padded = torch.cat([grad_output, padding], dim=1)
        else:
            grad_output_padded = grad_output
            
        # 将grad_output重塑为分组形式，与前向传播时保持一致
        grad_output_grouped = grad_output_padded.view(out_features, num_groups, group_size)
        # 扩展alpha维度以便广播
        alpha_expanded = alpha.unsqueeze(-1)
        
        # === 计算量化指示器 ===
        # 计算归一化后的权重值
        q_w = input_grouped / alpha_expanded
        # 小于量化下界的权重位置
        indicate_small = (q_w < Qn).float()
        # 大于量化上界的权重位置
        indicate_big = (q_w > Qp).float()
        # 在量化范围内的权重位置（这些位置会传递梯度）
        indicate_middle = 1.0 - indicate_small - indicate_big
        
        # === 计算缩放因子alpha的梯度 ===
        if ctx.num_bits == 1:
            # 1位量化：alpha的梯度与权重符号相关
            # sign()函数的导数在0处未定义，这里使用符号值作为近似
            grad_alpha = (input_grouped.sign() * grad_output_grouped * grad_scale).sum(dim=-1)
        else:
            # 多位量化：alpha的梯度包含三部分
            # 1. indicate_small * Qn: 小于下界的权重贡献量化下界的梯度
            # 2. indicate_big * Qp: 大于上界的权重贡献量化上界的梯度  
            # 3. indicate_middle * (-q_w + q_w.round()): 范围内权重的量化误差梯度
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + 
                          indicate_middle * (-q_w + q_w.round())) * 
                         grad_output_grouped * grad_scale).sum(dim=-1)
        
        # === 计算输入权重的梯度 ===
        # 使用直通估计器：只有在量化范围内的权重才传递梯度
        # 超出范围的权重梯度被截断为0
        grad_input_grouped = indicate_middle * grad_output_grouped
        
        # === 重塑梯度回原始形状 ===
        # 将分组形式的梯度重塑回 [out_features, padded_in_features]
        grad_input_padded = grad_input_grouped.view(out_features, padded_in_features)
        # 去掉填充部分，恢复到原始输入特征数
        grad_input = grad_input_padded[:, :in_features]
        
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
