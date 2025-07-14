# 分组Stretched Elastic量化的数学公式

## 1. 前向传播公式

### 1.1 基本参数设置
给定输入权重矩阵 $W \in \mathbb{R}^{H \times D}$，其中 $H$ 为输出特征维度，$D$ 为输入特征维度。

**分组参数：**
- 分组大小：$g$ (group_size)
- 分组数量：$G = \lceil D/g \rceil$
- 分组索引：$g_{idx}[i] = \lfloor i/g \rfloor$，其中 $i \in [0, D-1]$

**量化参数：**
- 量化位数：$b$ (num_bits)
- 裁剪值：$c = 1 - 10^{-2} = 0.99$
- 量化级别：$n = 2^{b-1}$ (当 $b > 0$ 时)
- 偏移值：$s = 0.5$ (当 $b > 0$ 时)
- 量化范围：$Q_p = \frac{n-s}{n}$，$Q_n = -Q_p$

### 1.2 分组量化过程

对于第 $k$ 组 ($k = 0, 1, \ldots, G-1$)：

**分组权重提取：**
$$W_k = W[:, kg:(k+1)g] \in \mathbb{R}^{H \times g_k}$$

其中 $g_k = \min(g, D - kg)$ 是第 $k$ 组的实际大小。

**缩放因子：**
$$\alpha_k \in \mathbb{R}^{H \times 1}$$

**量化过程：**

当 $b = 1$ 时（二值量化）：
$$\tilde{W}_k = \text{sign}(W_k)$$

当 $b > 1$ 时（Stretched Elastic量化）：
$$\bar{W}_k = \frac{W_k}{\alpha_k}$$
$$\hat{W}_k = \text{clamp}(\bar{W}_k, -c, c)$$
$$\tilde{W}_k = \frac{\text{round}(\hat{W}_k \cdot n - s) + s}{n}$$

**反量化：**
$$W_k^{(q)} = \tilde{W}_k \cdot \alpha_k$$

### 1.3 完整前向传播

$$W^{(q)} = [W_0^{(q)}, W_1^{(q)}, \ldots, W_{G-1}^{(q)}]$$

## 2. 反向传播公式

### 2.1 基本设置

**输入：**
- 上层梯度：$\frac{\partial L}{\partial W^{(q)}} \in \mathbb{R}^{H \times D}$

**输出：**
- 权重梯度：$\frac{\partial L}{\partial W}$
- 缩放因子梯度：$\frac{\partial L}{\partial \alpha}$

**梯度缩放：**
$$\gamma = \frac{1}{\sqrt{H \cdot D \cdot Q_p}}$$

### 2.2 量化指示器

对于第 $k$ 组，定义量化指示器：

$$q_k = \frac{W_k}{\alpha_k}$$

$$I_k^{small} = \mathbf{1}_{q_k < -c}$$
$$I_k^{big} = \mathbf{1}_{q_k > c}$$
$$I_k^{middle} = 1 - I_k^{small} - I_k^{big}$$

### 2.3 缩放因子梯度计算

当 $b = 1$ 时：
$$\frac{\partial L}{\partial \alpha_k} = \gamma \sum_{j=1}^{g_k} \left( \text{sign}(W_k) \odot \frac{\partial L}{\partial W_k^{(q)}} \right)$$

当 $b > 1$ 时：

首先计算量化后的归一化值：
$$\hat{q}_k = \text{clamp}(q_k, -c, c)$$
$$\tilde{q}_k = \frac{\text{round}(\hat{q}_k \cdot n - s) + s}{n}$$

然后计算梯度：
$$\frac{\partial L}{\partial \alpha_k} = \gamma \sum_{j=1}^{g_k} \left( \left( I_k^{small} \cdot Q_n + I_k^{big} \cdot Q_p + I_k^{middle} \cdot (-q_k + \tilde{q}_k) \right) \odot \frac{\partial L}{\partial W_k^{(q)}} \right)$$

### 2.4 权重梯度计算

使用直通估计器(Straight-Through Estimator)：

$$\frac{\partial L}{\partial W_k} = I_k^{middle} \odot \frac{\partial L}{\partial W_k^{(q)}}$$

### 2.5 完整反向传播

$$\frac{\partial L}{\partial W} = \left[ \frac{\partial L}{\partial W_0}, \frac{\partial L}{\partial W_1}, \ldots, \frac{\partial L}{\partial W_{G-1}} \right]$$

$$\frac{\partial L}{\partial \alpha} = \left[ \frac{\partial L}{\partial \alpha_0}, \frac{\partial L}{\partial \alpha_1}, \ldots, \frac{\partial L}{\partial \alpha_{G-1}} \right]$$

## 3. 关键特性

### 3.1 分组策略
- 采用与GPTQ相同的分组方式
- 按输入特征维度分组，每组包含连续的 $g$ 个特征
- 每组独立进行量化和缩放

### 3.2 Stretched Elastic量化特性
- 使用特殊的量化范围 $[-c, c]$ 而非标准的 $[Q_n, Q_p]$
- 采用非均匀的量化级别计算
- 在反向传播中使用修正的梯度计算公式

### 3.3 梯度传播机制
- 权重梯度：只有在量化范围内的权重才传递梯度
- 缩放因子梯度：综合考虑三个区域的贡献
- 使用梯度缩放因子 $\gamma$ 进行数值稳定性控制