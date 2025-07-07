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


class LsqGroupWiseExtension(torch.autograd.Function):
    """
    Group-wise quantization extension based on Learned Step-size Quantization.
    Modified from the original LSQ implementation to support group-wise quantization
    with optional asymmetric quantization (zero point).
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, group_size, dim=-1, zero_point=None):
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
            zero_point_reshaped, zero_point_expanded = _reshape_for_groupwise(input, zero_point, group_size, dim)
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
            return grad_output, None, None, None, None, None

        input_, alpha, zero_point = ctx.saved_tensors
        grad_scale, Qn, Qp, original_shape = ctx.other
        group_size = ctx.group_size
        dim = ctx.dim

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
            _, zero_point_expanded = _reshape_for_groupwise(input_, zero_point, group_size, dim)

        if ctx.asymmetric:
            q_w = (input_reshaped - zero_point_expanded) / alpha_expanded
        else:
            q_w = input_reshaped / alpha_expanded

        # Calculate indicators
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        # Calculate gradient w.r.t. alpha - Fixed per math formulas
        if ctx.num_bits == 1:
            if ctx.asymmetric:
                grad_alpha_term = ((input_reshaped >= zero_point_expanded).float()) * grad_output_reshaped * grad_scale
            else:
                grad_alpha_term = input_reshaped.sign() * grad_output_reshaped * grad_scale
        else:
            # Clamp q_w to prevent extreme values
            q_w_clamped = torch.clamp(q_w, -1e6, 1e6)
            q_w_rounded = q_w_clamped.round()
            
            if ctx.asymmetric:
                # For asymmetric quantization: q_w = (x - z) / alpha
                # Fixed per math doc: Use -Qp/2 instead of Qn=0 to prevent gradient vanishing
                grad_alpha_term = (
                    indicate_small * (-Qp / 2.0)  # Fixed: was Qn=0, now -Qp/2
                    + indicate_big * Qp
                    + indicate_middle * (-q_w_clamped + q_w_rounded)
                ) * grad_output_reshaped * grad_scale
            else:
                # For symmetric quantization: q_w = x / alpha  
                grad_alpha_term = (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_w_clamped + q_w_rounded)
                ) * grad_output_reshaped * grad_scale

        # Clamp gradient term to prevent overflow
        grad_alpha_term = torch.clamp(grad_alpha_term, -1e6, 1e6)

        # Sum over group dimension to get gradient for each group
        grad_alpha = _sum_over_groups(grad_alpha_term, group_size, dim, alpha.shape)
        
        # Additional gradient clipping for alpha
        grad_alpha = torch.clamp(grad_alpha, -1e3, 1e3)

        # Calculate gradient w.r.t. zero_point - Fixed per math formulas
        grad_zero_point = None
        if ctx.asymmetric:
            if ctx.num_bits == 1:
                # For 1-bit: w_q = I(x >= z) * alpha + z
                # ∂w_q/∂z = -δ(x - z) * alpha + 1 ≈ 1 (approximation for STE)
                grad_zero_point_term = grad_output_reshaped * grad_scale
            else:
                # Fixed per math doc: Complete zero point gradient with both contributions
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

        # Calculate gradient w.r.t. input - Fixed per math formulas
        if ctx.asymmetric:
            # Fixed per math doc: Use Qp/4 instead of Qn=0 to prevent gradient vanishing
            grad_input = (
                indicate_small * (-Qp )  # Fixed: was Qn=0, now Qp/4
                + indicate_middle * 1.0
                + indicate_big * Qp
            ) * grad_output_reshaped
        else:
            grad_input = (
                indicate_small * Qn
                + indicate_middle * 1.0
                + indicate_big * Qp
            ) * grad_output_reshaped
        
        grad_input = grad_input.view(original_shape)
        
        # Final NaN check
        if torch.any(torch.isnan(grad_alpha)):
            print("Warning: NaN detected in grad_alpha, setting to zero")
            grad_alpha = torch.zeros_like(grad_alpha)
        
        if grad_zero_point is not None and torch.any(torch.isnan(grad_zero_point)):
            print("Warning: NaN detected in grad_zero_point, setting to zero")
            grad_zero_point = torch.zeros_like(grad_zero_point)

        return grad_input, grad_alpha, None, None, None, grad_zero_point


class StretchedElasticGroupWiseQuant(torch.autograd.Function):
    """
    Group-wise Stretched Elastic Quantization with optional asymmetric quantization.
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, group_size, dim=-1, zero_point=None):
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

        # Calculate quantization parameters
        if ctx.asymmetric:
            # Asymmetric quantization
            if num_bits == 1:
                Qn, Qp = 0, 1
            else:
                Qn, Qp = 0, 2 ** num_bits - 1
        else:
            # Symmetric quantization
            if num_bits == 1 or num_bits == 0:
                Qn, Qp = -1, 1
            else:
                Qn, Qp = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1

        eps = torch.tensor(1e-5, device=alpha.device).float()  # Changed from 0.00001 to 1e-5 for consistency
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )

        # Reshape input and alpha for group-wise processing
        original_shape = input.shape
        input_reshaped, alpha_expanded = _reshape_for_groupwise(input, alpha, group_size, dim)

        # Handle zero point for asymmetric quantization
        zero_point_expanded = None
        if ctx.asymmetric:
            _, zero_point_expanded = _reshape_for_groupwise(input, zero_point, group_size, dim)
            zero_point_expanded = zero_point_expanded.detach()

        ctx.save_for_backward(input, alpha, zero_point)

        # Stretched elastic quantization parameters
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            if ctx.asymmetric:
                n_levels = 2 ** num_bits
                shift = 0
            else:
                n_levels = 2 ** (num_bits - 1)
                shift = 0.5

        if ctx.asymmetric:
            Qp_norm = (n_levels - 1) / n_levels
            Qn_norm = 0
        else:
            Qp_norm = (n_levels - shift) / n_levels
            Qn_norm = -Qp_norm

        ctx.other = grad_scale, Qn, Qp, original_shape, clip_val, n_levels, shift, Qn_norm, Qp_norm

        # Quantization
        if num_bits == 1:
            if ctx.asymmetric:
                q_w = (input_reshaped >= zero_point_expanded).float()
            else:
                q_w = input_reshaped.sign()
        else:
            if ctx.asymmetric:
                # Normalize input to [0, 1] range relative to zero_point
                normalized_input = (input_reshaped - zero_point_expanded) / alpha_expanded
                q_w = (
                    torch.round(
                        torch.clamp(normalized_input, 0, clip_val) * n_levels
                    )
                ) / n_levels
                w_q = q_w * alpha_expanded + zero_point_expanded
            else:
                # Symmetric case
                normalized_input = input_reshaped / alpha_expanded
                q_w = (
                    torch.round(
                        torch.clamp(normalized_input, -clip_val, clip_val) * n_levels - shift
                    )
                    + shift
                ) / n_levels
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
            return grad_output, None, None, None, None, None

        input_, alpha, zero_point = ctx.saved_tensors
        grad_scale, Qn, Qp, original_shape, clip_val, n_levels, shift, Qn_norm, Qp_norm = ctx.other
        group_size = ctx.group_size
        dim = ctx.dim

        # Reshape for group-wise processing
        input_reshaped, alpha_expanded = _reshape_for_groupwise(input_, alpha, group_size, dim)
        grad_output_reshaped = grad_output.view(input_reshaped.shape)

        # Handle zero point
        zero_point_expanded = None
        if ctx.asymmetric:
            _, zero_point_expanded = _reshape_for_groupwise(input_, zero_point, group_size, dim)

        if ctx.asymmetric:
            q_w = (input_reshaped - zero_point_expanded) / alpha_expanded
            clip_min, clip_max = 0, clip_val
        else:
            q_w = input_reshaped / alpha_expanded
            clip_min, clip_max = -clip_val, clip_val

        # Calculate indicators
        indicate_small = (q_w < clip_min).float()
        indicate_big = (q_w > clip_max).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        # Calculate gradient w.r.t. alpha - Fixed per math formulas
        if ctx.num_bits == 1:
            if ctx.asymmetric:
                grad_alpha_term = ((input_reshaped >= zero_point_expanded).float()) * grad_output_reshaped * grad_scale
            else:
                grad_alpha_term = input_reshaped.sign() * grad_output_reshaped * grad_scale
        else:
            # Complex quantization function for stretched elastic
            if ctx.asymmetric:
                quantized_term = (
                    torch.round(
                        torch.clamp(q_w, 0, clip_val) * n_levels
                    )
                ) / n_levels
            else:
                quantized_term = (
                    torch.round(
                        torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift
                    )
                    + shift
                ) / n_levels

            # Fixed per math doc: Use Qn_norm, but for asymmetric use modification for gradient vanishing
            if ctx.asymmetric:
                # Use modified gradient for small region to prevent vanishing
                grad_alpha_term = (
                    indicate_small * (-Qp_norm / 2.0)  # Fixed: use -Qp_norm/2 instead of Qn_norm=0
                    + indicate_big * Qp_norm
                    + indicate_middle * (-q_w + quantized_term)
                ) * grad_output_reshaped * grad_scale
            else:
                grad_alpha_term = (
                    indicate_small * Qn_norm
                    + indicate_big * Qp_norm
                    + indicate_middle * (-q_w + quantized_term)
                ) * grad_output_reshaped * grad_scale

        # Sum over group dimension to get gradient for each group
        grad_alpha = _sum_over_groups(grad_alpha_term, group_size, dim, alpha.shape)

        # Calculate gradient w.r.t. zero_point - Fixed per math formulas
        grad_zero_point = None
        if ctx.asymmetric:
            if ctx.num_bits == 1:
                # For 1-bit: w_q = I(x >= z) * alpha + z
                # ∂w_q/∂z = -δ(x - z) * alpha + 1 ≈ 1 (approximation for STE)
                grad_zero_point_term = grad_output_reshaped * grad_scale
            else:
                # Fixed per math doc: Complete zero point gradient with both contributions
                # Main contribution: ∂z/∂z = 1 (zero point directly adds to output)
                main_contribution = grad_output_reshaped * grad_scale
                
                # Additional contribution from quantization in middle region
                # In middle region: ∂(quantized_term * α)/∂z ≈ -1 (STE approximation)
                quantization_contribution = -indicate_middle * grad_output_reshaped * grad_scale
                
                grad_zero_point_term = main_contribution + quantization_contribution
                
            grad_zero_point = _sum_over_groups(grad_zero_point_term, group_size, dim, zero_point.shape)

        # Calculate gradient w.r.t. input - Fixed per math formulas
        if ctx.asymmetric:
            # Fixed per math doc: Use Qp_norm/4 instead of Qn_norm=0 to prevent gradient vanishing
            grad_input = (
                indicate_small * (-Qp_norm / 4.0)  # Fixed: was Qn_norm=0, now Qp_norm/4
                + indicate_middle * 1.0
                + indicate_big * Qp_norm
            ) * grad_output_reshaped
        else:
            grad_input = (
                indicate_small * Qn_norm
                + indicate_middle * 1.0
                + indicate_big * Qp_norm
            ) * grad_output_reshaped
        
        grad_input = grad_input.view(original_shape)

        return grad_input, grad_alpha, None, None, None, grad_zero_point


def _reshape_for_groupwise(input_tensor, alpha, group_size, dim):
    """
    Reshape input tensor and alpha for group-wise processing.
    For QAT: input is weight tensor [in_features, out_features], 
    alpha is [in_features, num_groups] where num_groups = ceil(out_features / group_size)
    
    :param input_tensor: input tensor to be reshaped [in_features, out_features]
    :param alpha: alpha tensor (per group) [in_features, num_groups]
    :param group_size: size of each group
    :param dim: dimension along which to group (should be -1 for out_features)
    :return: reshaped input and expanded alpha
    """
    # Handle negative dimension
    if dim < 0:
        dim = input_tensor.dim() + dim
    
    # For QAT weight quantization, we expect:
    # input_tensor: [in_features, out_features]
    # alpha: [in_features, num_groups]
    # dim: -1 (grouping along out_features dimension)
    
    if input_tensor.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {input_tensor.dim()}D tensor")
    
    if dim != 1:  # dim=1 corresponds to out_features dimension
        raise ValueError(f"For weight quantization, grouping should be along out_features dimension (dim=1), got dim={dim}")
    
    in_features, out_features = input_tensor.shape
    num_groups = (out_features + group_size - 1) // group_size
    
    # Verify alpha shape
    expected_alpha_shape = (in_features, num_groups)
    if alpha.shape != expected_alpha_shape:
        raise ValueError(f"Expected alpha shape {expected_alpha_shape}, got {alpha.shape}")
    
    # Pad out_features dimension if necessary
    if out_features % group_size != 0:
        pad_size = num_groups * group_size - out_features
        # Pad on the right side of out_features dimension
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_size))
    
    # Reshape input: [in_features, out_features] -> [in_features, num_groups, group_size]
    input_reshaped = input_tensor.view(in_features, num_groups, group_size)
    
    # Expand alpha: [in_features, num_groups] -> [in_features, num_groups, group_size]
    alpha_expanded = alpha.unsqueeze(-1).expand(in_features, num_groups, group_size)
    
    return input_reshaped, alpha_expanded


def _sum_over_groups(grad_tensor, group_size, dim, alpha_shape):
    """
    Sum gradients over groups to get per-group gradients.
    For QAT: grad_tensor has shape [in_features, num_groups, group_size]
    Need to sum over group_size dimension to get [in_features, num_groups]
    
    :param grad_tensor: gradient tensor [in_features, num_groups, group_size]
    :param group_size: size of each group
    :param dim: dimension along which groups are organized (should be 1 for out_features)
    :param alpha_shape: target shape for alpha gradients [in_features, num_groups]
    :return: summed gradients per group [in_features, num_groups]
    """
    # For the corrected reshape, grad_tensor has shape [in_features, num_groups, group_size]
    # We need to sum over the group_size dimension (last dimension)
    grad_alpha = grad_tensor.sum(dim=-1)  # Sum over group_size dimension
    
    # grad_alpha now has shape [in_features, num_groups] which matches alpha_shape
    return grad_alpha


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
            # === 使用分组量化（与GPTQ分组方式一致）===
            if self.w_bits == 2 or self.w_bits == 0:
                # 使用分组Stretched Elastic量化
                weight = StretchedElasticGroupWiseQuant.apply(
                    real_weights,
                    self.scale,
                    self.w_bits,
                    self.group_size,
                    -1,
                    self.zero if not self.sym else None
                ).to(input_.dtype)
            elif self.w_bits <= 4:
                # 使用分组LSQ量化
                weight = LsqGroupWiseExtension.apply(
                    real_weights,
                    self.scale,
                    self.w_bits,
                    self.group_size,
                    -1,
                    self.zero if not self.sym else None,
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
        
        # 执行线性变换
        out = nn.functional.linear(input_, weight)
        
        # 添加偏置（如果有的话）
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
            
        return out
    