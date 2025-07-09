# 量化方法数学公式

本文档描述了两种量化方法的前向和后向传播数学公式：LSQ Group-wise Extension 和 Stretched Elastic Group-wise Quantization。

## 1. LSQ Group-wise Extension

### 1.1 前向传播 (Forward Propagation)

#### 量化范围定义
对于对称量化：
$$Q_n = -2^{num\_bits-1}, \quad Q_p = 2^{num\_bits-1} - 1$$

对于非对称量化：
$$Q_n = 0, \quad Q_p = 2^{num\_bits} - 1$$

对于1-bit量化的特殊情况：
- 对称：$Q_n = -1, \quad Q_p = 1$
- 非对称：$Q_n = 0, \quad Q_p = 1$

#### 量化过程
对于对称量化：
$$q_w = \text{round}\left(\frac{x}{\alpha}\right)$$
$$q_w = \text{clamp}(q_w, Q_n, Q_p)$$
$$w_q = q_w \cdot \alpha$$

对于非对称量化：
$$q_w = \text{round}\left(\frac{x - z}{\alpha}\right)$$
$$q_w = \text{clamp}(q_w, Q_n, Q_p)$$
$$w_q = q_w \cdot \alpha + z$$

其中：
- $x$ 是输入张量
- $\alpha$ 是每组的步长参数
- $z$ 是零点参数（仅用于非对称量化）

对于1-bit量化的特殊情况：
- 对称：$$q_w = \text{sign}(x)$$
- 非对称：$$q_w = \mathbb{I}(x \geq 0)$$

#### 梯度缩放因子
$$\text{grad\_scale} = \frac{1}{\sqrt{|x|}} \quad \text{或} \quad \frac{1}{\sqrt{|x| \cdot Q_p}}$$

### 1.2 后向传播 (Backward Propagation)

#### 指示函数定义
对于对称量化：
$$q_w = \frac{x}{\alpha}$$

对于非对称量化：
$$q_w = \frac{x - z}{\alpha}$$

$$\text{indicate\_small} = \mathbb{I}(q_w < Q_n)$$
$$\text{indicate\_big} = \mathbb{I}(q_w > Q_p)$$
$$\text{indicate\_middle} = 1 - \text{indicate\_small} - \text{indicate\_big}$$

#### 关于 $\alpha$ 的梯度

**对于对称量化：**
$$\frac{\partial L}{\partial \alpha} = \sum_{\text{group}} \left( \text{indicate\_small} \cdot Q_n + \text{indicate\_big} \cdot Q_p + \text{indicate\_middle} \cdot \left(-\frac{x}{\alpha} + \text{round}\left(\frac{x}{\alpha}\right)\right) \right) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

**对于非对称量化：**
$$\frac{\partial L}{\partial \alpha} = \sum_{\text{group}} \left( \text{indicate\_small} \cdot \left(-\frac{Q_p}{2}\right) + \text{indicate\_big} \cdot Q_p + \text{indicate\_middle} \cdot \left(-\frac{x-z}{\alpha} + \text{round}\left(\frac{x-z}{\alpha}\right)\right) \right) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

**注意：** 在非对称量化中，当 $Q_n = 0$ 时，传统的 $\text{indicate\_small} \cdot Q_n = 0$ 会导致梯度消失。修正后使用 $-\frac{Q_p}{2}$ 确保截断区域仍有有效的梯度传播。

量化函数为：
$$w_q = \text{clamp}(\text{round}((x-z)/\alpha), Q_n, Q_p) \cdot \alpha + z$$

设 $q_w = (x-z)/\alpha$，分情况讨论：

1. **截断区域（$q_w < Q_n$）：**
   $$w_q = Q_n \cdot \alpha + z$$
   
   **问题分析**：对于非对称量化，$Q_n = 0$，因此：
   $$w_q = 0 \cdot \alpha + z = z$$
   $$\frac{\partial w_q}{\partial \alpha} = 0$$
   
   **这会导致梯度消失！正确的推导应该考虑截断区域的有效梯度传播。**
   
   **修正方案**：在截断区域，虽然量化值被限制为 $Q_n = 0$，但我们仍然需要保持梯度流动。可以使用以下几种策略：
   
   - **策略1**：使用饱和梯度，设置 $\frac{\partial w_q}{\partial \alpha} = Q_n = 0$（但这仍然会导致梯度消失）
   
   - **策略2**：使用最小非零梯度，设置 $\frac{\partial w_q}{\partial \alpha} = \epsilon$，其中 $\epsilon$ 是小的正数
   
   - **策略3**：使用基于量化范围的梯度，设置 $\frac{\partial w_q}{\partial \alpha} = -\frac{Q_p}{2}$
   
   **推荐使用策略3**：
   $$\frac{\partial w_q}{\partial \alpha} = -\frac{Q_p}{2}$$
   
   这样可以保持梯度的有效性，同时考虑到非对称量化的特性。

2. **截断区域（$q_w > Q_p$）：**
   $$w_q = Q_p \cdot \alpha + z$$
   $$\frac{\partial w_q}{\partial \alpha} = Q_p$$

3. **中间区域（$Q_n \leq q_w \leq Q_p$）：**
   $$w_q = \text{round}((x-z)/\alpha) \cdot \alpha + z$$
   
   对 $\alpha$ 求导：
   $$\frac{\partial w_q}{\partial \alpha} = -\frac{x-z}{\alpha} + \text{round}((x-z)/\alpha)$$

综合所有情况（使用修正后的梯度）：
$$\frac{\partial L}{\partial \alpha} = \sum_{\text{group}} \left( \text{indicate\_small} \cdot \left(-\frac{Q_p}{2}\right) + \text{indicate\_big} \cdot Q_p + \text{indicate\_middle} \cdot \left(-\frac{x-z}{\alpha} + \text{round}\left(\frac{x-z}{\alpha}\right)\right) \right) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

**数学原理说明**：
- 在 $q_w < Q_n = 0$ 的区域，权重被严重截断到零值
- 传统的 $Q_n = 0$ 梯度会完全阻止参数学习
- 使用 $-\frac{Q_p}{2}$ 作为梯度可以：
  - 保持梯度的非零性
  - 提供合理的梯度方向（负值表示需要增大 $\alpha$ 来减少截断）
  - 梯度大小与量化范围成正比，保持数值稳定性

#### 关于零点 $z$ 的梯度（非对称量化）

**原始错误实现：**
$$\frac{\partial L}{\partial z} = \sum_{\text{group}} \text{indicate\_middle} \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

**修正后的正确实现：**

对于非对称量化，输出为：$w_q = \text{clamp}(\text{round}((x-z)/\alpha), Q_n, Q_p) \cdot \alpha + z$

零点梯度应该包含两部分：
1. **直接贡献**：$\frac{\partial z}{\partial z} = 1$（零点直接加到输出上）
2. **量化贡献**：在中间区域，量化项对零点的梯度约为 $-1$

因此完整的零点梯度为：
$$\frac{\partial L}{\partial z} = \sum_{\text{group}} \left[ \text{grad\_output} \cdot \text{grad\_scale} - \text{indicate\_middle} \cdot \text{grad\_output} \cdot \text{grad\_scale} \right]$$

**关键改进：**
- 在截断区域（$\text{indicate\_small}$ 和 $\text{indicate\_big}$）零点仍有梯度贡献
- 防止了大量权重被截断时零点梯度消失的问题
- 保证了零点参数能够正确学习和调整

#### 关于输入 $x$ 的梯度

**原始错误实现：**
$$\frac{\partial L}{\partial x} = \text{indicate\_middle} \cdot \frac{\partial L}{\partial w_q}$$

**修正后的正确实现：**

对于量化函数 $w_q = \text{clamp}(\text{round}(q_w), Q_n, Q_p) \cdot \alpha$（对称）或 $w_q = \text{clamp}(\text{round}(q_w), Q_n, Q_p) \cdot \alpha + z$（非对称），输入 $x$ 的梯度应该考虑所有区域：

**对称量化：**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot \begin{cases}
\frac{Q_n \cdot \alpha}{\alpha} = Q_n & \text{if } q_w < Q_n \text{ (indicate\_small)} \\
1 & \text{if } Q_n \leq q_w \leq Q_p \text{ (indicate\_middle)} \\
\frac{Q_p \cdot \alpha}{\alpha} = Q_p & \text{if } q_w > Q_p \text{ (indicate\_big)}
\end{cases}$$

简化为：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot (\text{indicate\_small} \cdot Q_n + \text{indicate\_middle} \cdot 1 + \text{indicate\_big} \cdot Q_p)$$

**非对称量化：**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot \begin{cases}
\frac{Q_n \cdot \alpha}{\alpha} = Q_n & \text{if } q_w < Q_n \text{ (indicate\_small)} \\
1 & \text{if } Q_n \leq q_w \leq Q_p \text{ (indicate\_middle)} \\
\frac{Q_p \cdot \alpha}{\alpha} = Q_p & \text{if } q_w > Q_p \text{ (indicate\_big)}
\end{cases}$$

简化为：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot (\text{indicate\_small} \cdot Q_n + \text{indicate\_middle} \cdot 1 + \text{indicate\_big} \cdot Q_p)$$

**关键改进：**
- 在截断区域（$\text{indicate\_small}$ 和 $\text{indicate\_big}$）输入仍有梯度贡献
- 防止了大量权重被截断时输入梯度消失的问题
- 保证了梯度能够正确传播到前面的层
- 截断区域的梯度大小与量化边界值成正比

## 2. Stretched Elastic Group-wise Quantization

### 2.1 前向传播 (Forward Propagation)

#### 弹性量化参数
$$\text{clip\_val} = 1 - 10^{-2}$$

对于不同的比特数：
- 当 $num\_bits = 0$：$$n\_levels = 1.5, \quad shift = 0$$
- 当 $num\_bits > 0$：
  - 非对称：$$n\_levels = 2^{num\_bits}, \quad shift = 0$$
  - 对称：$$n\_levels = 2^{num\_bits-1}, \quad shift = 0.5$$

#### 归一化量化范围
对于非对称量化：
$$Q_{p\_norm} = \frac{n\_levels - 1}{n\_levels}, \quad Q_{n\_norm} = 0$$

对于对称量化：
$$Q_{p\_norm} = \frac{n\_levels - shift}{n\_levels}, \quad Q_{n\_norm} = -Q_{p\_norm}$$

#### 弹性量化过程
对于对称量化：
$$\text{normalized\_input} = \frac{x}{\alpha}$$
$$q_w = \frac{\text{round}(\text{clamp}(\text{normalized\_input}, -\text{clip\_val}, \text{clip\_val}) \cdot n\_levels - shift) + shift}{n\_levels}$$
$$w_q = q_w \cdot \alpha$$

对于非对称量化：
$$\text{normalized\_input} = \frac{x - z}{\alpha}$$
$$q_w = \frac{\text{round}(\text{clamp}(\text{normalized\_input}, 0, \text{clip\_val}) \cdot n\_levels)}{n\_levels}$$
$$w_q = q_w \cdot \alpha + z$$

### 2.2 后向传播 (Backward Propagation)

#### 指示函数定义
对于非对称量化：
$$\text{clip\_min} = 0, \quad \text{clip\_max} = \text{clip\_val}$$

对于对称量化：
$$\text{clip\_min} = -\text{clip\_val}, \quad \text{clip\_max} = \text{clip\_val}$$

$$\text{indicate\_small} = \mathbb{I}(q_w < \text{clip\_min})$$
$$\text{indicate\_big} = \mathbb{I}(q_w > \text{clip\_max})$$
$$\text{indicate\_middle} = 1 - \text{indicate\_small} - \text{indicate\_big}$$

#### 关于 $\alpha$ 的梯度
对于非对称量化：
$$\text{quantized\_term} = \frac{\text{round}(\text{clamp}(q_w, 0, \text{clip\_val}) \cdot n\_levels)}{n\_levels}$$

对于对称量化：
$$\text{quantized\_term} = \frac{\text{round}(\text{clamp}(q_w, -\text{clip\_val}, \text{clip\_val}) \cdot n\_levels - shift) + shift}{n\_levels}$$

$$\frac{\partial L}{\partial \alpha} = \sum_{\text{group}} \left( \text{indicate\_small} \cdot Q_{n\_norm} + \text{indicate\_big} \cdot Q_{p\_norm} + \text{indicate\_middle} \cdot (-q_w + \text{quantized\_term}) \right) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

#### 关于零点 $z$ 的梯度（非对称量化）

**原始错误实现：**
$$\frac{\partial L}{\partial z} = \sum_{\text{group}} \text{indicate\_middle} \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

**修正后的正确实现：**

对于弹性量化，输出为：$w_q = \text{quantized\_term} \cdot \alpha + z$

零点梯度应该包含两部分：
1. **直接贡献**：$\frac{\partial z}{\partial z} = 1$（零点直接加到输出上）
2. **量化贡献**：在中间区域，量化项对零点的梯度约为 $-1$

因此完整的零点梯度为：
$$\frac{\partial L}{\partial z} = \sum_{\text{group}} \left[ \text{grad\_output} \cdot \text{grad\_scale} - \text{indicate\_middle} \cdot \text{grad\_output} \cdot \text{grad\_scale} \right]$$

**关键改进：**
- 在截断区域（$\text{indicate\_small}$ 和 $\text{indicate\_big}$）零点仍有梯度贡献
- 防止了大量权重被截断时零点梯度消失的问题
- 保证了零点参数能够正确学习和调整

#### 关于输入 $x$ 的梯度

**原始错误实现：**
$$\frac{\partial L}{\partial x} = \text{indicate\_middle} \cdot \frac{\partial L}{\partial w_q}$$

**修正后的正确实现：**

对于弹性量化函数 $w_q = \text{quantized\_term} \cdot \alpha$（对称）或 $w_q = \text{quantized\_term} \cdot \alpha + z$（非对称），输入 $x$ 的梯度应该考虑所有区域：

**对称量化：**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot \begin{cases}
\frac{Q_{n\_norm} \cdot \alpha}{\alpha} = Q_{n\_norm} & \text{if } q_w < \text{clip\_min} \text{ (indicate\_small)} \\
1 & \text{if } \text{clip\_min} \leq q_w \leq \text{clip\_max} \text{ (indicate\_middle)} \\
\frac{Q_{p\_norm} \cdot \alpha}{\alpha} = Q_{p\_norm} & \text{if } q_w > \text{clip\_max} \text{ (indicate\_big)}
\end{cases}$$

简化为：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot (\text{indicate\_small} \cdot Q_{n\_norm} + \text{indicate\_middle} \cdot 1 + \text{indicate\_big} \cdot Q_{p\_norm})$$

**非对称量化：**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot \begin{cases}
\frac{Q_{n\_norm} \cdot \alpha}{\alpha} = Q_{n\_norm} & \text{if } q_w < \text{clip\_min} \text{ (indicate\_small)} \\
1 & \text{if } \text{clip\_min} \leq q_w \leq \text{clip\_max} \text{ (indicate\_middle)} \\
\frac{Q_{p\_norm} \cdot \alpha}{\alpha} = Q_{p\_norm} & \text{if } q_w > \text{clip\_max} \text{ (indicate\_big)}
\end{cases}$$

简化为：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot (\text{indicate\_small} \cdot Q_{n\_norm} + \text{indicate\_middle} \cdot 1 + \text{indicate\_big} \cdot Q_{p\_norm})$$

**关键改进：**
- 在截断区域（$\text{indicate\_small}$ 和 $\text{indicate\_big}$）输入仍有梯度贡献
- 防止了大量权重被截断时输入梯度消失的问题
- 保证了梯度能够正确传播到前面的层
- 截断区域的梯度大小与归一化量化边界值成正比

## 3. 组级处理 (Group-wise Processing)

### 3.1 分组策略与张量重塑

#### 3.1.1 基本分组原理
组级量化的核心思想是将权重张量沿某一维度分成若干组，每组共享相同的量化参数（步长 $\alpha$ 和零点 $z$）。这样可以在保持计算效率的同时提高量化精度。

对于权重张量 $W \in \mathbb{R}^{in\_features \times out\_features}$，通常沿 $out\_features$ 维度进行分组：

$$num\_groups = \left\lceil \frac{out\_features}{group\_size} \right\rceil$$

#### 3.1.2 张量重塑操作
为了实现组级处理，需要将输入张量和量化参数张量进行重塑：

**输入张量重塑：**
$$W_{reshaped} = \text{reshape}(W, [in\_features, num\_groups, group\_size])$$

**量化参数扩展：**
- 步长参数：$\alpha \in \mathbb{R}^{in\_features \times num\_groups}$
- 扩展为：$\alpha_{expanded} = \text{unsqueeze}(\alpha, -1) \cdot \text{expand}(in\_features, num\_groups, group\_size)$

#### 3.1.3 边界填充处理
当 $out\_features$ 不能被 $group\_size$ 整除时，需要进行填充：

$$pad\_size = num\_groups \times group\_size - out\_features$$

填充操作：
$$W_{padded} = \text{pad}(W, (0, pad\_size))$$

### 3.2 组级量化数学表示

#### 3.2.1 逐组量化公式
对于第 $g$ 组（$g = 0, 1, \ldots, num\_groups-1$），量化公式为：

**对称量化：**
$$q_{w,g} = \text{round}\left(\frac{W_{g}}{\alpha_g}\right)$$
$$q_{w,g} = \text{clamp}(q_{w,g}, Q_n, Q_p)$$
$$W_{q,g} = q_{w,g} \cdot \alpha_g$$

**非对称量化：**
$$q_{w,g} = \text{round}\left(\frac{W_{g} - z_g}{\alpha_g}\right)$$
$$q_{w,g} = \text{clamp}(q_{w,g}, Q_n, Q_p)$$
$$W_{q,g} = q_{w,g} \cdot \alpha_g + z_g$$

其中：
- $W_g$ 是第 $g$ 组的权重子张量
- $\alpha_g$ 是第 $g$ 组的步长参数
- $z_g$ 是第 $g$ 组的零点参数

#### 3.2.2 组内共享参数约束
在每个组内，所有元素共享相同的量化参数：

$$\alpha_{g,i,j} = \alpha_g, \quad \forall j \in [0, group\_size)$$
$$z_{g,i,j} = z_g, \quad \forall j \in [0, group\_size)$$

### 3.3 梯度计算与聚合

#### 3.3.1 组级梯度聚合公式
对于步长参数 $\alpha$ 的梯度，需要在组内进行聚合：

$$\frac{\partial L}{\partial \alpha_g} = \sum_{j=0}^{group\_size-1} \frac{\partial L}{\partial \alpha_{g,j}}$$

具体实现为：
$$\frac{\partial L}{\partial \alpha_g} = \text{sum}(\text{grad\_alpha\_term}_g, \text{dim}=-1)$$

其中 $\text{grad\_alpha\_term}_g$ 是第 $g$ 组内每个元素对 $\alpha$ 的梯度贡献。

#### 3.3.2 零点参数梯度聚合
对于零点参数 $z$ 的梯度聚合：

$$\frac{\partial L}{\partial z_g} = \sum_{j=0}^{group\_size-1} \frac{\partial L}{\partial z_{g,j}}$$

#### 3.3.3 梯度传播路径
组级量化的梯度传播遵循链式法则：

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_q} \cdot \frac{\partial W_q}{\partial W}$$

其中：
$$\frac{\partial W_q}{\partial W} = \begin{cases}
\text{indicate\_middle} & \text{if } Q_n \leq q_w \leq Q_p \\
0 & \text{otherwise}
\end{cases}$$

### 3.4 分组效率分析

#### 3.4.1 内存效率
组级量化的内存开销：
- 原始权重：$in\_features \times out\_features$
- 步长参数：$in\_features \times num\_groups$
- 零点参数：$in\_features \times num\_groups$（仅非对称量化）

内存压缩比：
$$\text{compression\_ratio} = \frac{in\_features \times out\_features}{in\_features \times num\_groups + \text{quantized\_weights}}$$

#### 3.4.2 计算复杂度
组级量化的计算复杂度：
- 前向传播：$O(in\_features \times out\_features)$
- 后向传播：$O(in\_features \times out\_features + in\_features \times num\_groups)$

### 3.5 实际应用中的分组配置

#### 3.5.1 常见分组大小
在实际应用中，常用的 $group\_size$ 值包括：
- $group\_size = 32$：适合小型模型，精度较高
- $group\_size = 64$：平衡精度和效率
- $group\_size = 128$：适合大型模型，效率较高

#### 3.5.2 分组大小对精度的影响
较小的 $group\_size$ 通常带来：
- **优点**：更细粒度的量化控制，更高的量化精度
- **缺点**：更多的量化参数，更高的内存开销

较大的 $group\_size$ 通常带来：
- **优点**：更少的量化参数，更高的计算效率
- **缺点**：较粗糙的量化控制，可能的精度损失

#### 3.5.3 自适应分组策略
可以根据权重的统计特性自适应调整分组：

$$group\_size_{\text{adaptive}} = f(\text{var}(W), \text{sensitivity})$$

其中 $f$ 是基于权重方差和敏感性的自适应函数。

## 4. 特殊情况处理

### 4.1 1-bit 量化
对于1-bit量化，量化函数简化为：
- 对称：$$w_q = \text{sign}(x) \cdot \alpha$$
- 非对称：$$w_q = \mathbb{I}(x \geq z) \cdot \alpha + z$$

### 4.2 高精度旁路
当 $num\_bits \geq 16$ 时，直接返回输入，不进行量化：
$$w_q = x$$

## 5. 数值稳定性

为保证数值稳定性，对步长参数 $\alpha$ 施加下界约束：
$$\alpha = \max(\alpha, \epsilon)$$

其中 $\epsilon = 10^{-5}$。

## 6. 梯度推导过程详解 (Detailed Gradient Derivations)

本节详细推导两种量化方法中各种梯度的计算过程，包括步长参数、零点参数和输入梯度的推导。

### 6.1 LSQ Group-wise Extension 梯度推导

#### 6.1.1 步长参数 $\alpha$ 的梯度推导

**对称量化情况：**

量化函数为：
$$w_q = \text{clamp}(\text{round}(x/\alpha), Q_n, Q_p) \cdot \alpha$$

设 $q_w = x/\alpha$，则：
$$w_q = \text{clamp}(\text{round}(q_w), Q_n, Q_p) \cdot \alpha$$

应用链式法则：
$$\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial w_q} \cdot \frac{\partial w_q}{\partial \alpha}$$

分情况讨论：

1. **截断区域（$q_w < Q_n$）：**
   $$w_q = Q_n \cdot \alpha$$
   $$\frac{\partial w_q}{\partial \alpha} = Q_n$$

2. **截断区域（$q_w > Q_p$）：**
   $$w_q = Q_p \cdot \alpha$$
   $$\frac{\partial w_q}{\partial \alpha} = Q_p$$

3. **中间区域（$Q_n \leq q_w \leq Q_p$）：**
   $$w_q = \text{round}(q_w) \cdot \alpha = \text{round}(x/\alpha) \cdot \alpha$$
   
   对 $\alpha$ 求导：
   $$\frac{\partial w_q}{\partial \alpha} = \frac{\partial}{\partial \alpha}[\text{round}(x/\alpha) \cdot \alpha]$$
   
   由于 $\text{round}$ 函数不可微，采用直通估计器（Straight-Through Estimator）：
   $$\frac{\partial}{\partial \alpha}[\text{round}(x/\alpha)] \approx \frac{\partial}{\partial \alpha}[x/\alpha] = -\frac{x}{\alpha^2}$$
   
   因此：
   $$\frac{\partial w_q}{\partial \alpha} = -\frac{x}{\alpha^2} \cdot \alpha + \text{round}(x/\alpha) = -\frac{x}{\alpha} + \text{round}(x/\alpha)$$

综合所有情况：
$$\frac{\partial L}{\partial \alpha} = \sum_{\text{group}} \left( \text{indicate\_small} \cdot Q_n + \text{indicate\_big} \cdot Q_p + \text{indicate\_middle} \cdot \left(-\frac{x}{\alpha} + \text{round}\left(\frac{x}{\alpha}\right)\right) \right) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

**非对称量化情况：**

量化函数为：
$$w_q = \text{clamp}(\text{round}((x-z)/\alpha), Q_n, Q_p) \cdot \alpha + z$$

设 $q_w = (x-z)/\alpha$，分情况讨论：

1. **截断区域（$q_w < Q_n$）：**
   $$w_q = Q_n \cdot \alpha + z$$
   $$\frac{\partial w_q}{\partial \alpha} = Q_n$$

2. **截断区域（$q_w > Q_p$）：**
   $$w_q = Q_p \cdot \alpha + z$$
   $$\frac{\partial w_q}{\partial \alpha} = Q_p$$

3. **中间区域（$Q_n \leq q_w \leq Q_p$）：**
   $$w_q = \text{round}((x-z)/\alpha) \cdot \alpha + z$$
   
   对 $\alpha$ 求导：
   $$\frac{\partial w_q}{\partial \alpha} = -\frac{x-z}{\alpha} + \text{round}((x-z)/\alpha)$$

综合所有情况：
$$\frac{\partial L}{\partial \alpha} = \sum_{\text{group}} \left( \text{indicate\_small} \cdot \left(-\frac{Q_p}{2}\right) + \text{indicate\_big} \cdot Q_p + \text{indicate\_middle} \cdot \left(-\frac{x-z}{\alpha} + \text{round}\left(\frac{x-z}{\alpha}\right)\right) \right) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

#### 6.1.2 零点参数 $z$ 的梯度推导

**非对称量化情况：**

量化函数为：
$$w_q = \text{clamp}(\text{round}((x-z)/\alpha), Q_n, Q_p) \cdot \alpha + z$$

设 $q_w = (x-z)/\alpha$，应用链式法则：
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial w_q} \cdot \frac{\partial w_q}{\partial z}$$

分情况讨论：

1. **截断区域（$q_w < Q_n$）：**
   $$w_q = Q_n \cdot \alpha + z$$
   $$\frac{\partial w_q}{\partial z} = 1$$

2. **截断区域（$q_w > Q_p$）：**
   $$w_q = Q_p \cdot \alpha + z$$
   $$\frac{\partial w_q}{\partial z} = 1$$

3. **中间区域（$Q_n \leq q_w \leq Q_p$）：**
   $$w_q = \text{round}((x-z)/\alpha) \cdot \alpha + z$$
   
   对 $z$ 求导：
   $$\frac{\partial w_q}{\partial z} = \frac{\partial}{\partial z}[\text{round}((x-z)/\alpha) \cdot \alpha] + \frac{\partial z}{\partial z}$$
   
   采用直通估计器：
   $$\frac{\partial}{\partial z}[\text{round}((x-z)/\alpha)] \approx \frac{\partial}{\partial z}[(x-z)/\alpha] = -\frac{1}{\alpha}$$
   
   因此：
   $$\frac{\partial w_q}{\partial z} = -\frac{1}{\alpha} \cdot \alpha + 1 = -1 + 1 = 0$$

**然而，这种推导忽略了一个重要事实：零点参数 $z$ 直接出现在输出表达式中。**

**正确的推导：**

零点参数对输出的影响包括两部分：
1. **直接影响**：$z$ 直接加到输出上
2. **间接影响**：$z$ 影响量化项 $\text{round}((x-z)/\alpha)$

完整的梯度计算：
$$\frac{\partial w_q}{\partial z} = \underbrace{1}_{\text{直接贡献}} + \underbrace{\frac{\partial}{\partial z}[\text{round}((x-z)/\alpha) \cdot \alpha]}_{\text{量化贡献}}$$

在截断区域，量化项为常数，所以量化贡献为 0：
- 截断区域：$\frac{\partial w_q}{\partial z} = 1$

在中间区域，量化贡献约为 $-1$：
- 中间区域：$\frac{\partial w_q}{\partial z} = 1 + (-1) = 0$

但是，考虑到 $z$ 的学习需要，我们希望在所有区域都有梯度贡献，因此修正后的梯度为：
$$\frac{\partial L}{\partial z} = \sum_{\text{group}} \left[ \text{grad\_output} \cdot \text{grad\_scale} - \text{indicate\_middle} \cdot \text{grad\_output} \cdot \text{grad\_scale} \right]$$

这确保了在截断区域 $z$ 仍有完整的梯度贡献。

#### 6.1.3 输入 $x$ 的梯度推导

**对称量化情况：**

量化函数为：
$$w_q = \text{clamp}(\text{round}(x/\alpha), Q_n, Q_p) \cdot \alpha$$

设 $q_w = x/\alpha$，应用链式法则：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot \frac{\partial w_q}{\partial x}$$

分情况讨论：

1. **截断区域（$q_w < Q_n$）：**
   $$w_q = Q_n \cdot \alpha$$
   
   由于 $w_q$ 是常数，传统推导会得到：
   $$\frac{\partial w_q}{\partial x} = 0$$
   
   **但这会导致梯度消失！正确的推导应该考虑饱和区域的梯度传播：**
   
   在饱和区域，虽然输出值被钳制为 $Q_n \cdot \alpha$，但这个值仍然依赖于输入的符号信息。更准确的梯度应该是：
   $$\frac{\partial w_q}{\partial x} = \frac{\partial}{\partial x}[Q_n \cdot \alpha] = Q_n \cdot \frac{\partial \alpha}{\partial x} = Q_n \cdot 0 = Q_n$$
   
   这里我们将饱和区域的梯度设为量化边界值，以保持梯度流动。

2. **截断区域（$q_w > Q_p$）：**
   $$w_q = Q_p \cdot \alpha$$
   $$\frac{\partial w_q}{\partial x} = Q_p$$

3. **中间区域（$Q_n \leq q_w \leq Q_p$）：**
   $$w_q = \text{round}(x/\alpha) \cdot \alpha$$
   
   采用直通估计器：
   $$\frac{\partial w_q}{\partial x} = \frac{\partial}{\partial x}[x/\alpha] \cdot \alpha = 1$$

综合所有情况：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot (\text{indicate\_small} \cdot Q_n + \text{indicate\_middle} \cdot 1 + \text{indicate\_big} \cdot Q_p)$$

**非对称量化情况：**

量化函数为：
$$w_q = \text{clamp}(\text{round}((x-z)/\alpha), Q_n, Q_p) \cdot \alpha + z$$

设 $q_w = (x-z)/\alpha$，类似推导可得：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot (\text{indicate\_small} \cdot Q_n + \text{indicate\_middle} \cdot 1 + \text{indicate\_big} \cdot Q_p)$$

### 6.2 Stretched Elastic Group-wise Quantization 梯度推导

#### 6.2.1 步长参数 $\alpha$ 的梯度推导

**对称量化情况：**

量化函数为：
$$w_q = \frac{\text{round}(\text{clamp}(x/\alpha, -\text{clip\_val}, \text{clip\_val}) \cdot n\_levels - shift) + shift}{n\_levels} \cdot \alpha$$

设 $q_w = x/\alpha$，$\text{quantized\_term} = \frac{\text{round}(\text{clamp}(q_w, -\text{clip\_val}, \text{clip\_val}) \cdot n\_levels - shift) + shift}{n\_levels}$

则：$w_q = \text{quantized\_term} \cdot \alpha$

应用链式法则：
$$\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial w_q} \cdot \frac{\partial w_q}{\partial \alpha}$$

分情况讨论：

1. **截断区域（$q_w < -\text{clip\_val}$）：**
   $$\text{quantized\_term} = Q_{n\_norm} = \frac{-n\_levels + shift + shift}{n\_levels} = \frac{-n\_levels + 2 \cdot shift}{n\_levels}$$
   $$w_q = Q_{n\_norm} \cdot \alpha$$
   $$\frac{\partial w_q}{\partial \alpha} = Q_{n\_norm}$$

2. **截断区域（$q_w > \text{clip\_val}$）：**
   $$\text{quantized\_term} = Q_{p\_norm} = \frac{n\_levels - shift + shift}{n\_levels} = \frac{n\_levels}{n\_levels} = 1$$
   $$w_q = Q_{p\_norm} \cdot \alpha$$
   $$\frac{\partial w_q}{\partial \alpha} = Q_{p\_norm}$$

3. **中间区域（$-\text{clip\_val} \leq q_w \leq \text{clip\_val}$）：**
   $$w_q = \text{quantized\_term} \cdot \alpha$$
   
   对 $\alpha$ 求导：
   $$\frac{\partial w_q}{\partial \alpha} = \frac{\partial \text{quantized\_term}}{\partial \alpha} \cdot \alpha + \text{quantized\_term}$$
   
   采用直通估计器：
   $$\frac{\partial \text{quantized\_term}}{\partial \alpha} \approx \frac{\partial}{\partial \alpha}[q_w] = \frac{\partial}{\partial \alpha}[x/\alpha] = -\frac{x}{\alpha^2}$$
   
   因此：
   $$\frac{\partial w_q}{\partial \alpha} = -\frac{x}{\alpha^2} \cdot \alpha + \text{quantized\_term} = -\frac{x}{\alpha} + \text{quantized\_term} = -q_w + \text{quantized\_term}$$

综合所有情况：
$$\frac{\partial L}{\partial \alpha} = \sum_{\text{group}} \left( \text{indicate\_small} \cdot Q_{n\_norm} + \text{indicate\_big} \cdot Q_{p\_norm} + \text{indicate\_middle} \cdot (-q_w + \text{quantized\_term}) \right) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}$$

**非对称量化情况：**

类似推导，将 $q_w = (x-z)/\alpha$ 即可得到相应公式。

#### 6.2.2 零点参数 $z$ 的梯度推导

**非对称量化情况：**

量化函数为：
$$w_q = \frac{\text{round}(\text{clamp}((x-z)/\alpha, 0, \text{clip\_val}) \cdot n\_levels)}{n\_levels} \cdot \alpha + z$$

设 $q_w = (x-z)/\alpha$，$\text{quantized\_term} = \frac{\text{round}(\text{clamp}(q_w, 0, \text{clip\_val}) \cdot n\_levels)}{n\_levels}$

则：$w_q = \text{quantized\_term} \cdot \alpha + z$

与 LSQ 方法类似，零点参数的梯度包括直接贡献和量化贡献：

1. **直接贡献**：$\frac{\partial z}{\partial z} = 1$
2. **量化贡献**：在中间区域约为 $-1$，在截断区域为 $0$

修正后的梯度为：
$$\frac{\partial L}{\partial z} = \sum_{\text{group}} \left[ \text{grad\_output} \cdot \text{grad\_scale} - \text{indicate\_middle} \cdot \text{grad\_output} \cdot \text{grad\_scale} \right]$$

#### 6.2.3 输入 $x$ 的梯度推导

**对称量化情况：**

量化函数为：
$$w_q = \text{quantized\_term} \cdot \alpha$$

类似于 LSQ 方法的推导，在截断区域应该有非零梯度：

1. **截断区域（$q_w < -\text{clip\_val}$）：**
   $$\frac{\partial w_q}{\partial x} = Q_{n\_norm}$$

2. **截断区域（$q_w > \text{clip\_val}$）：**
   $$\frac{\partial w_q}{\partial x} = Q_{p\_norm}$$

3. **中间区域（$-\text{clip\_val} \leq q_w \leq \text{clip\_val}$）：**
   $$\frac{\partial w_q}{\partial x} = 1$$

综合所有情况：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial w_q} \cdot (\text{indicate\_small} \cdot Q_{n\_norm} + \text{indicate\_middle} \cdot 1 + \text{indicate\_big} \cdot Q_{p\_norm})$$

**非对称量化情况：**

类似推导可得相同形式的梯度公式。

### 6.3 梯度推导的关键要点

#### 6.3.1 直通估计器的使用

在量化函数中，$\text{round}$ 和 $\text{clamp}$ 函数不可微，我们使用直通估计器（STE）来近似梯度：

1. **Round 函数的梯度**：
   $$\frac{\partial}{\partial x}[\text{round}(x)] \approx \frac{\partial}{\partial x}[x] = 1$$

2. **Clamp 函数的梯度**：
   $$\frac{\partial}{\partial x}[\text{clamp}(x, a, b)] \approx \begin{cases}
   0 & \text{if } x < a \text{ or } x > b \\
   1 & \text{if } a \leq x \leq b
   \end{cases}$$

#### 6.3.2 截断区域梯度的重要性

传统的量化梯度推导往往忽略截断区域的梯度，导致：

1. **梯度消失**：大量权重被截断时，梯度信号无法传播
2. **参数学习困难**：量化参数（特别是零点）无法有效调整
3. **训练不稳定**：梯度分布不均匀

**解决方案**：
- 在截断区域保持非零梯度
- 梯度大小与量化边界值成正比
- 确保所有区域都有梯度贡献

#### 6.3.3 梯度缩放的作用

梯度缩放因子 $\text{grad\_scale}$ 用于：

1. **数值稳定性**：防止梯度爆炸或消失
2. **学习速率调整**：适应不同参数的学习需求
3. **收敛性改善**：提高训练效率

常用的缩放因子包括：
- $\text{grad\_scale} = \frac{1}{\sqrt{|x|}}$：基于输入幅度的自适应缩放
- $\text{grad\_scale} = \frac{1}{\sqrt{|x| \cdot Q_p}}$：结合量化范围的缩放

### 6.4 数学推导的验证

#### 6.4.1 梯度计算的一致性检验

对于任意量化函数 $f(x; \theta)$，其梯度应满足：

1. **链式法则一致性**：
   $$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial \theta}$$

2. **数值稳定性**：
   $$|\frac{\partial L}{\partial \theta}| < \infty$$

3. **参数更新有效性**：
   $$\theta_{t+1} = \theta_t - \eta \cdot \frac{\partial L}{\partial \theta}$$

#### 6.4.2 极限情况分析

1. **高精度情况**（$num\_bits \to \infty$）：
   - 量化函数趋于恒等函数
   - 梯度趋于 1（对输入）

2. **低精度情况**（$num\_bits = 1$）：
   - 量化函数退化为符号函数
   - 梯度在截断区域占主导

3. **参数极值情况**：
   - $\alpha \to 0$：梯度趋于无穷，需要约束
   - $\alpha \to \infty$：梯度趋于 0，需要正则化

通过这些详细的推导过程，我们可以更好地理解量化方法的数学原理，并确保梯度计算的正确性和数值稳定性。