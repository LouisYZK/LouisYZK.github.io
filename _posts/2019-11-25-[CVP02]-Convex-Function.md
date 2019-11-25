---
layout:     post
date:       2019-11-25
tag:        note
author:     BY Zhi-kai Yang
---

> Covex optitimization第二周，凸函数。xjtu的课程老师更侧重凸函数作为函数一般的性质，例如连续性、可微的一些证明，讲了更多利用上镜图研究凸函数的一些知识；
>
> Boyd课程涵盖更多凸函数的保凸运算和凸函数性质的例子。本文结合两部分摘选一部分记录。

## Convex Function

### 1. Definition

$f:\mathbb{R^n} \rightarrow \mathbb{R}$ 是凸函数，且$dom\ f $是凸集，如果
$$
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y),\ \forall x,y \in \boldsymbol{dom} f
$$
则称$f$是其定义域上的凸函数。如果$\leq$为$\lt$ 成为严格凸；

**另一个等价条件：**
$$
g(t) = f(x + tv),\\ \{t|x+tv \in dom\ f\}
$$
$g(t)$为凸函数；这个等价条件可以用来将高维函数低维处理来证明凸性；

### 2. 一阶条件FOC

如果函数一节可微，凸函数的一阶条件为：
$$
\forall x,y \in dom\ f, \ f(y) \ge f(x) + \nabla^Tf(y-x)
$$

### 3.二阶条件SOC

如果函数二阶在定义域内处处二阶可微分，则凸函数的二阶条件为：
$$
\forall x \in dom\ f, \nabla^2f(x) \ge0
$$
也就当是处处的Hessian矩阵半正定（一阶导数不减）；

### 上镜图与凸函数的连续性

$$
epi(f) = \{(x,w)|x\in C, w\in R, f(x) \le w \}
$$

实值函数扩展前后的上镜图不变，$dom(f)$是$epi(f)$在$\mathbb{R^n}$的投影；

**实值函数为凸函数**  $\iff$ 上镜图为凸集

上镜图为**闭集**的函数称之为**闭函数**

在跳跃点处取值较小的函数值的函数称之为**下半连续的**

则下面三个条件的是等价的：

- 任意水平集合是闭的；
- 函数下半连续
- 函数上镜图是闭集

### 4. Examples

$e^{ax},\ \forall a \in R, x\in R$ 为凸；

$x^a,\ x \in \mathbb{R_{++}},\ a\ge 1\ or\ a\le 0$为凸

$|x|^p,\ p\ge1$  为凸

$xlogx,\ x\in \mathbb{R_{++}}$ 为凸

所有的Norm （范数）都是凸函数（除了0范数，因为0范数不是范数）

Log-Sum-Exp: $log\sum{e^{x_i}}$ 他是max函数的一个解析近似；

几何平均数： $f(x) = (\prod_{i=1}^N\ x_i)^{1/N}$, 是**凹函数**， $domf = \mathbb{R_{++}^n}$

对数绝对值函数： $f(x) = log\ det(X)$,   $domf = \mathbb{S_{++}^n} $ (可以用第二个等价条件切换到一维证明)

**Jessen不等式**， 如果函数是凸的，则在期望上有如下结论：
$$
f(E(x)) \le E(f(x))
$$

## 保持凸性的函数运算

### 非负加权

有限：
$$
f = w_1f_1 + w_2f_2 + ... + w_nf_n
$$
无限：
$$
g(x) = \int_{y\in A} f(x, y)w(y)dy
$$

### 仿射组合

$$
g(x) = f(Ax +b)\\
dom\ g = \{x | Ax+b \in dom\ f \}
$$

### Pointwise 最大

$$
f(x) = max\{f_1(x), f_2(x), ..., f_n(x)\} \\
dom\ f = dom\ f_1 \cap ...dom\ f_n
$$

无限维：
$$
g(x) = \sup_{y\in A} f(x, y)
$$
例如：
$$
max\ \{a_1^Tx +b_1, ...., a_n^Tx+b_n \}
$$
是多个仿射组合的取最大，保持凸性；

### 函数组合

我们定义 函数组合 
$$
f = h(g(x)),\\
dom\ f = \{x\in dom\ g|g(x) \in dom\ h  \}
$$
若考虑组合都是标量函数的情况下，假设二阶可微，求出二阶条件：
$$
f^{''} (x) = h^{''}(g(x))g^{'}(x)^2 + h^{'}(x)g^{''}(x)
$$
可以得出如下的一些保凸条件：

- $\tilde{h}$ 为凸且不减， $g$为凸
- $\tilde{h}$为凸且不增，$g$为凹

### Minimization

如果$f(x,y)$是凸函数，则
$$
g(x)  =\inf_{y\in C} f(x, y)
$$
是凸函数；

### 透视函数

$$
g(x, t) = tf(\frac{x}{t})
$$

若$f$为凸，则透视函数$g$为凸；

例子：

$f(x) = -logx$为凸， 则$g(x, t) = tlog\frac{t}{x}$为凸，这个函数可以用来证明**KL散度**总是凸函数

### 共轭函数

无论函数$f$是否是凸函数，他的共轭函数
$$
f^*(y) = \sup_{x\in dom\ f} (x^Ty - f(x))
$$
都为凸函数；

共轭函数可以从图像上看出是一系列线性函数取最大；Boyd讲课时举了一个经济学中的现实意义；如果$x$是生产品数量，$y$是市场价格，$f(x)$是成本函数，则成本函数的共轭就是每种市场条件下生产得到的最大利润值；

例子： 

$f(x) = x^TQx$ 的共轭是 $f^*(y) = y^TQ^{-1}y$

## 拟凸函数与拟凹函数

### Definition

从两个方面来定义：

1. 所有的低水平集为凸集，则函数为拟凸，凸函数一定是拟凸函数；

2. 满足下列：

$$
f(\theta x + (1-\theta)y) \le max\ \{f(x),f (y)\}
$$

可见拟凸函数要求的**凸组合的函数值**只需要小于两端中较大值即可；

### FOC

函数拟凸时，他有下列的一个逻辑关系：
$$
f(y) \le f(x) \Longrightarrow \ \nabla^Tf(x)(y-x) \le 0
$$

### SOC

函数拟凸且二阶可微时，存在如下的逻辑关系：
$$
y^T \nabla f(x) = 0 \Longrightarrow y^T \nabla^2f(x)y \ge 0
$$
即拟凸函数不需要处处二阶矩阵半正定；

## 对数凹

几乎所有的分布函数都是对数凹的；



## Reference

- [Standfold CVX 02](http://web.stanford.edu/class/ee364a/lectures/functions.pdf)
- Convex Optimization, Stephen Boyd
- [中科大 凌青 最优化理论](https://www.bilibili.com/video/av29071445/?p=19&t=2639)


> 哦，天呐。为什么这星期字数变少了？莫非是沉迷炉石自走棋不能自拔？ 啊，其实是这样的。凸函数这章节其实有很多有意思的案例证明。有很多与其他知识联系的函数的凸性证明，但是这些证明都在Boyd的原作中能找到；唯一需要记录的是xjtu课堂上关于凸函数连续性的一些证明，但是我没有泛函的基础，很多还没看太明白，等我研究好了再回来补充。
