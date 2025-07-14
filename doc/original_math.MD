# 量化函数的数学形式

## 1. LsqBinaryTernaryExtension 类

### 1.1 前向传播

#### 参数定义
- `input`: 输入张量 $x$
- `alpha`: 步长参数 $\alpha$
- `num_bits`: 量化位数 $b$
- `layerwise`: 是否按层量化

#### 量化范围计算
$$
Q_n, Q_p = \begin{cases}
-1, 1 & \text{if } b = 1 \text{ or } b = 0 \\
-2^{b-1}, 2^{b-1} - 1 & \text{otherwise}
\end{cases}
$$

#### 步长保护
$$
\alpha = \max(\alpha, \epsilon), \quad \epsilon = 10^{-5}
$$

#### 梯度缩放因子
$$
\text{grad\_scale} = \begin{cases}
\frac{1}{\sqrt{|x|}} & \text{if } Q_p = 0 \\
\frac{1}{\sqrt{|x| \cdot Q_p}} & \text{otherwise}
\end{cases}
$$

#### 量化过程
$$
q_w = \begin{cases}
\text{sign}(x) & \text{if } b = 1 \\
\text{clamp}(\text{round}(\frac{x}{\alpha}), Q_n, Q_p) & \text{otherwise}
\end{cases}
$$

#### 输出
$$
w_q = q_w \cdot \alpha
$$

### 1.2 反向传播

#### 指示函数
$$
\begin{align}
I_{\text{small}} &= \mathbb{1}_{q_w < Q_n} \\
I_{\text{big}} &= \mathbb{1}_{q_w > Q_p} \\
I_{\text{middle}} &= 1 - I_{\text{small}} - I_{\text{big}}
\end{align}
$$

其中 $q_w = \frac{x}{\alpha}$

#### 关于 $\alpha$ 的梯度

**对于 $b = 1$ (二值化):**
$$
\frac{\partial L}{\partial \alpha} = \begin{cases}
\sum(\text{sign}(x) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{if layerwise} \\
\sum_{\text{dim}=-1}(\text{sign}(x) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{otherwise}
\end{cases}
$$

**对于 $b > 1$:**
$$
\frac{\partial L}{\partial \alpha} = \begin{cases}
\sum((I_{\text{small}} \cdot Q_n + I_{\text{big}} \cdot Q_p + I_{\text{middle}} \cdot (-q_w + \text{round}(q_w))) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{if layerwise} \\
\sum_{\text{dim}=-1}((I_{\text{small}} \cdot Q_n + I_{\text{big}} \cdot Q_p + I_{\text{middle}} \cdot (-q_w + \text{round}(q_w))) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{otherwise}
\end{cases}
$$

#### 关于输入的梯度
$$
\frac{\partial L}{\partial x} = I_{\text{middle}} \cdot \frac{\partial L}{\partial w_q}
$$

## 2. StretchedElasticQuant 类

### 2.1 前向传播

#### 参数定义
- `input`: 输入张量 $x$
- `alpha`: 步长参数 $\alpha$
- `num_bits`: 量化位数 $b$
- `layerwise`: 是否按层量化

#### 量化参数计算
$$
\begin{align}
\text{clip\_val} &= 1 - 10^{-2} \\
n_{\text{levels}}, \text{shift} &= \begin{cases}
1.5, 0 & \text{if } b = 0 \\
2^{b-1}, 0.5 & \text{otherwise}
\end{cases} \\
Q_p &= \frac{n_{\text{levels}} - \text{shift}}{n_{\text{levels}}} \\
Q_n &= -Q_p
\end{align}
$$

#### 量化过程
$$
q_w = \begin{cases}
\text{sign}(x) & \text{if } b = 1 \\
\frac{\text{round}(\text{clamp}(\frac{x}{\alpha}, -\text{clip\_val}, \text{clip\_val}) \cdot n_{\text{levels}} - \text{shift}) + \text{shift}}{n_{\text{levels}}} & \text{otherwise}
\end{cases}
$$

#### 输出
$$
w_q = q_w \cdot \alpha
$$

### 2.2 反向传播

#### 指示函数
$$
\begin{align}
I_{\text{small}} &= \mathbb{1}_{q_w < -\text{clip\_val}} \\
I_{\text{big}} &= \mathbb{1}_{q_w > \text{clip\_val}} \\
I_{\text{middle}} &= 1 - I_{\text{small}} - I_{\text{big}}
\end{align}
$$

其中 $q_w = \frac{x}{\alpha}$

#### 关于 $\alpha$ 的梯度

**对于 $b = 1$ (二值化):**
$\frac{\partial L}{\partial \alpha} = \begin{cases}
\sum(\text{sign}(x) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{if layerwise} \\
\sum_{\text{dim}=-1}(\text{sign}(x) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{otherwise}
\end{cases}$
##

**对于 $b > 1$:**

定义量化函数：
$$f(q_w) = \frac{\text{round}(\text{clamp}(q_w, -\text{clip\_val}, \text{clip\_val}) \cdot n_{\text{levels}} - \text{shift}) + \text{shift}}{n_{\text{levels}}}$$

则梯度为：
$$\frac{\partial L}{\partial \alpha} = \begin{cases}
\sum((I_{\text{small}} \cdot Q_n + I_{\text{big}} \cdot Q_p + I_{\text{middle}} \cdot (-q_w + f(q_w))) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{if layerwise} \\
\sum_{\text{dim}=-1}((I_{\text{small}} \cdot Q_n + I_{\text{big}} \cdot Q_p + I_{\text{middle}} \cdot (-q_w + f(q_w))) \cdot \frac{\partial L}{\partial w_q} \cdot \text{grad\_scale}) & \text{otherwise}
\end{cases}$$

#### 关于输入的梯度
$\frac{\partial L}{\partial x} = I_{\text{middle}} \cdot \frac{\partial L}{\partial w_q}$

## 3. 主要差异

### 3.1 量化范围定义
- **LsqBinaryTernaryExtension**: 使用传统的对称量化范围
- **StretchedElasticQuant**: 使用拉伸的弹性量化范围，引入了 `clip_val` 和更复杂的level计算

### 3.2 量化函数
- **LsqBinaryTernaryExtension**: 简单的四舍五入并夹紧到量化范围
- **StretchedElasticQuant**: 使用拉伸变换，先缩放再偏移，然后量化

### 3.3 梯度计算
- 两者在计算 $\alpha$ 的梯度时使用相似的结构，但 **StretchedElasticQuant** 在中间区域使用更复杂的量化函数导数

### 3.4 数值稳定性
- 两者都使用了 `eps` 来避免除零错误
- **StretchedElasticQuant** 额外使用了 `clip_val` 来限制输入范围，提高数值稳定性