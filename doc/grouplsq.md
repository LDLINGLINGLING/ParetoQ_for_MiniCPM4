# 分组LSQ量化的数学表达式

## 1. 概述

分组LSQ（Learned Step-size Quantization）量化是LSQ算法的扩展版本，它将权重矩阵按列分组，每组使用独立的量化缩放因子。这种方法能够更好地适应权重分布的差异，提高量化精度。

## 2. 符号定义

- $W \in \mathbb{R}^{O \times I}$：输入权重矩阵，其中 $O$ 为输出特征数，$I$ 为输入特征数
- $\alpha \in \mathbb{R}^{O \times G}$：每组的缩放因子，$G$ 为组数
- $z \in \mathbb{R}^{O \times G}$：每组的零点（可选）
- $b$：量化位数
- $g$：组大小（group_size）
- $G = \lceil I/g \rceil$：组数
- $Q_n, Q_p$：量化范围的下界和上界

## 3. 量化范围定义

量化范围根据位数确定：

$$
\begin{cases}
Q_n = -1, Q_p = 1 & \text{if } b = 1 \text{ (二值量化)} \\
Q_n = -2^{b-1}, Q_p = 2^{b-1} - 1 & \text{if } b > 1 \text{ (多位量化)}
\end{cases}
$$

## 4. 前向传播

### 4.1 分组重塑

首先将权重矩阵重塑为分组形式：

$$
W_{grouped} = \text{reshape}(W, (O, G, g))
$$

其中如果 $I$ 不能被 $g$ 整除，需要进行零填充。

### 4.2 量化过程

#### 二值量化 ($b = 1$)

$$
\begin{align}
Q_W &= \text{sign}(W_{grouped}) \\
\tilde{W} &= Q_W \odot \alpha
\end{align}
$$

其中 $\odot$ 表示逐元素乘法，$\alpha$ 被扩展到与 $W_{grouped}$ 相同的维度。

#### 多位量化 ($b > 1$)

$$
\begin{align}
\hat{W} &= \frac{W_{grouped} - z}{\alpha} \\
Q_W &= \text{clamp}(\text{round}(\hat{W}), Q_n, Q_p) \\
\tilde{W} &= Q_W \odot \alpha + z
\end{align}
$$

### 4.3 梯度缩放因子

根据LSQ论文，梯度缩放因子定义为：

$$
s = \frac{1}{\sqrt{g \cdot O \cdot Q_p}}
$$

## 5. 反向传播

### 5.1 梯度计算准备

设 $\frac{\partial L}{\partial \tilde{W}}$ 为从后续层传回的梯度。

#### 二值量化梯度

对于二值量化，梯度计算如下：

$$
\begin{align}
\frac{\partial L}{\partial \alpha} &= s \cdot \sum_{i=1}^{g} \left( \text{sign}(W_{grouped}) \odot \frac{\partial L}{\partial \tilde{W}} \right) \\
\frac{\partial L}{\partial W} &= \frac{\partial L}{\partial \tilde{W}}
\end{align}
$$

#### 多位量化梯度

对于多位量化，首先定义指示器函数：

$$
\begin{align}
I_{small} &= \mathbb{1}[\hat{W} < Q_n] \\
I_{big} &= \mathbb{1}[\hat{W} > Q_p] \\
I_{middle} &= 1 - I_{small} - I_{big}
\end{align}
$$

其中 $\mathbb{1}[\cdot]$ 为指示函数。

### 5.2 各参数梯度

#### 缩放因子梯度

$$
\frac{\partial L}{\partial \alpha} = s \cdot \sum_{i=1}^{g} \left[ \left( I_{small} \cdot Q_n + I_{big} \cdot Q_p + I_{middle} \cdot (-\hat{W} + \text{round}(\hat{W})) \right) \odot \frac{\partial L}{\partial \tilde{W}} \right]
$$

#### 输入权重梯度（直通估计器）

$$
\frac{\partial L}{\partial W} = I_{middle} \odot \frac{\partial L}{\partial \tilde{W}}
$$

#### 零点梯度

如果使用零点参数：

$$
\frac{\partial L}{\partial z} = s \cdot \sum_{i=1}^{g} \left[ (I_{small} + I_{big} + I_{middle}) \odot \frac{\partial L}{\partial \tilde{W}} \right]
$$

## 6. 关键数学性质

### 6.1 直通估计器（STE）

对于量化操作中的不可微分部分，使用直通估计器：

$$
\frac{\partial \text{round}(x)}{\partial x} \approx 1 \quad \text{for } x \in [Q_n, Q_p]
$$

### 6.2 梯度缩放

梯度缩放因子 $s$ 的作用是平衡不同参数的梯度幅度，确保训练稳定性。它基于权重矩阵的维度和量化范围进行归一化。

### 6.3 分组独立性

每个组的量化参数 $\alpha$ 和 $z$ 是独立优化的，这使得算法能够适应不同组内权重分布的差异。

## 7. 实现细节

### 7.1 数值稳定性

为避免除零错误，缩放因子被限制在最小值之上：

$$
\alpha = \max(\alpha, \epsilon)
$$

其中 $\epsilon = 10^{-5}$。

### 7.2 边界情况处理

- 当 $b \geq 16$ 时，不执行量化，直接返回原始权重
- 当输入特征数不能被组大小整除时，使用零填充
- 对输入中的NaN值进行检测和处理

## 8. 算法复杂度

- **时间复杂度**：$O(O \times I)$，与权重矩阵大小成线性关系
- **空间复杂度**：$O(O \times G)$，额外存储每组的量化参数
- **内存效率**：通过分组减少了量化参数的数量，相比逐元素量化更加内存友好

## 9. 优势分析

1. **适应性强**：每组独立的缩放因子能够适应不同的权重分布
2. **精度提升**：相比全局量化，分组量化通常能获得更高的精度
3. **计算效率**：向量化操作提高了计算效率
4. **可扩展性**：支持不同的量化位数和组大小配置