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
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, group_size):
        """
        :param input: 输入权重张量，形状为 [out_features, in_features]
        :param alpha: 每组的缩放因子，形状为 [out_features, num_groups]
        :param num_bits: 量化位数
        :param group_size: 每组的大小
        :return: 分组量化后的权重
        """
        ctx.num_bits = num_bits
        ctx.group_size = group_size
        
        if num_bits >= 16:
            return input
            
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1
            
        eps = torch.tensor(1e-5, device=alpha.device, dtype=alpha.dtype)
        alpha = torch.where(alpha > eps, alpha, eps)
        
        out_features, in_features = input.shape
        num_groups = (in_features + group_size - 1) // group_size
        
        # 将输入重塑为分组形式
        padded_in_features = num_groups * group_size
        if in_features != padded_in_features:
            # 如果需要，填充输入特征
            padding = torch.zeros(out_features, padded_in_features - in_features, 
                                device=input.device, dtype=input.dtype)
            input_padded = torch.cat([input, padding], dim=1)
        else:
            input_padded = input
            
        # 重塑为 [out_features, num_groups, group_size]
        input_grouped = input_padded.view(out_features, num_groups, group_size)
        
        # 扩展alpha维度以匹配分组形状
        alpha_expanded = alpha.unsqueeze(-1)  # [out_features, num_groups, 1]
        
        grad_scale = 1.0 / math.sqrt(input.numel() * Qp) if Qp else 1.0 / math.sqrt(input.numel())
        
        ctx.save_for_backward(input_grouped, alpha)
        ctx.other = grad_scale, Qn, Qp, out_features, in_features
        
        if num_bits == 1:
            q_w = input_grouped.sign()
        else:
            q_w = (input_grouped / alpha_expanded).round().clamp(Qn, Qp)
            
        w_q_grouped = q_w * alpha_expanded
        
        # 重塑回原始形状
        w_q_padded = w_q_grouped.view(out_features, padded_in_features)
        w_q = w_q_padded[:, :in_features]
        
        return w_q
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None
            
        input_grouped, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, out_features, in_features = ctx.other
        
        # 处理grad_output的形状
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
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        
        if ctx.num_bits == 1:
            grad_alpha = (input_grouped.sign() * grad_output_grouped * grad_scale).sum(dim=-1)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + 
                          indicate_middle * (-q_w + q_w.round())) * 
                         grad_output_grouped * grad_scale).sum(dim=-1)
        
        grad_input_grouped = indicate_middle * grad_output_grouped
        grad_input_padded = grad_input_grouped.view(out_features, padded_in_features)
        grad_input = grad_input_padded[:, :in_features]
        
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
