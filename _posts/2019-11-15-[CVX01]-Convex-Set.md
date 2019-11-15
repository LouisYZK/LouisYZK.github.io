---
layout:     post
date:       2019-11-15
tag:        note
author:     BY Zhi-kai Yang
---

>  这是XJTU秋下学期《凸分析与优化》课程的笔记，共32课时。概念比较多，复杂且易混淆，我放弃尝试英文记录。(因为我发现数学严格定义的语序上英文总是与中文相反.....)
>
>  内容除了老师使用的的中文slides外，部分不严格的定义参考了斯坦福CVX的slides; 同时老师对一些定理做了推导证明；

## Convex Set

**definition:** 令集合$C \subseteq \mathbb{R}^{n}$ ,对于$\forall x, y \in C$ 和$\forall \alpha \in [0,1]$ 如果有 $$\alpha \boldsymbol{x}+(1-\alpha) \boldsymbol{y} \in C$$ ，则称$C$为凸集。约定空集是凸集。

**example:** 

- 超平面 
- 球 $B(\boldsymbol{z}, \delta)=\left\{x \in \mathbb{R}^{n} |\|\boldsymbol{x}-\boldsymbol{z}\|<\delta\right\}$
- 正定矩阵、半正定矩阵、对称矩阵

**保持凸性的运算：**($C1$和$C2$均为凸集)

- $C_{1} \cap C_{2}$ 和 $C_1 + C_2$ (向量和)
- 任意scalar $\lambda$,  $\lambda C$是凸集
- 闭包$cl(C)$ 和内部$int(C)$ 都是凸集
- 放射变换下的像和原像
  - $\text { suppose } f: \mathbf{R}^{n} \rightarrow \mathbf{R}^{m} \text { is affine }\left(f(x)=A x+b \text { with } A \in \mathbf{R}^{m \times n}, b \in \mathbf{R}^{m}\right)$
  - $S \subseteq \mathbf{R}^{n} \text { convex } \Longrightarrow f(S)=\{f(x) | x \in S\} \text { convex }$
  - $C \subseteq \mathbf{R}^{m} \text { convex } \quad \Longrightarrow \quad f^{-1}(C)=\left\{x \in \mathbf{R}^{n} | f(x) \in C\right\} \text { convex }$

**泛函概念补充(此课程默认prerequisite是泛函分析？)**

> - 内点： 对于度量空间$(X, d)$ , $M \subset X$ , 如果$x \in M$, $\exist r > 0$, 若$B(x, r) \subset M$, 则$x$为$M$的内点；（一般的度量空间需要满足一些性质，比一般的向量空间更加宽泛，严格定义参看泛函中的定义）
> - 内部：$M$中所有的内点集合为内部，$int(M)$
> - 开集：如果$M = int(M)$ , $M$是开集
> - 如果 $X / M$ 差集为开集，则$M$是闭集
> - 聚点（还有别的翻译）:  如果$x\in M$, $M \cap (B(x, r) / {x}) \neq \emptyset$ 则$x$为$M$的聚点；
> - 闭包：所有聚合点与内点的并集即为$M$的闭包；那么闭集就等价于闭包； 

## Convex Combination and Convex Hull

**definition:** 

- 集合$C \subseteq \mathbb{R^n}$, 则称$\mathbb{R^n}$中所有包含$C$的凸集的交集为$C$的**凸包**；记做$conv(C)$
- $x^{1}, \ldots, x^{m} \in C, \alpha_{1}, \ldots, \alpha_{m} \geq 0, \sum_{i=1}^{m} \alpha_{i}=1$ ,称$\boldsymbol{y}=\sum_{i=1}^{m} \alpha_{i} \boldsymbol{x}^{i}$ 为$C$中$m$个向量的凸组合；

可以用凸组合来定义凸包：

$conv(C)$中任意向量均可表示为$C$中有限个向量的凸组合，既对任意$\forall x \in C$有：$\boldsymbol{x}=\sum_{i=1}^{m} w_{i} \boldsymbol{x}_{i}$ ,$\boldsymbol{x}_{i} \in C, w_{i} \geq 0, i=1, \ldots, m$, $\sum_{i=1}^{m} w_{i}=1$

**properties** of Convec Hull:

- $C$ 是凸集，$C = conv(C)$
- $x \in conv(C)$, 则$\operatorname{conv}(C \cup\{x\})=\operatorname{conv}(C)$
- $C_1, C_2 \subseteq \mathbb{R^n}$ 则$\operatorname{conv}\left(C_{1}+C_{2}\right)=\operatorname{conv}\left(C_{1}\right)+\operatorname{conv}\left(C_{2}\right)$ 

## Affine Set \ Combination \ Hull

**definition:** 

- $x^{1}, \ldots, x^{m} \in C, \alpha_{1}, \ldots, \alpha_{m} \in \mathbb{R}, \sum_{i=1}^{m} \alpha_{i}=1$, $\sum_{i=1}^{m} \alpha_{i}=1$ 称$\boldsymbol{y}=\sum_{i=1}^{m} \alpha_{i} \boldsymbol{x}^{i}$ 为**仿射组合**；（与凸组合的区别在于去掉了[0, 1]的限制）
- 对于$\forall x,y \in C$ 和$\forall \alpha \in \mathbb{R}$ , 如果有$\alpha \boldsymbol{x}+(1-\alpha) \boldsymbol{y} \in C$ 成立，则称$C$为**仿射集**.
- $\mathbb{R^n}$中所有包含$C$的仿射集的交集称为$C$ 的**仿射包**，记做$aff(C)$

**Theorem:**

$aff(C)$中任意向量均可以表示为$C$ 中有限个向量的仿射组合。

**example:** 仿射集

- 实数空间$\mathbb{R^n}$中的点线、超平面
- 矩阵的零空间 $\operatorname{null}(A)=\left\{\boldsymbol{x} \in \mathbb{R}^{n} | A \boldsymbol{x}=0\right\}$

**properties** of affine hull

对任何集合$C$. 必然存在 
$$
\operatorname{aff}(C)=\operatorname{aff}(\operatorname{conv}(C))=\operatorname{aff}(\operatorname{cl}(C))
$$

> **proof**（作业，自己的证明，待严格验证）
>
> （1）证明$aff(C) = aff(conv(C))$
>
> ​	a. 如果$C$是凸集，则，$C=conv(C)$ 两个集合的仿射包自然相等；
>
> ​    b. 如果C不为凸集，$\forall x \in conv(C)$ 有
> $$
> x = \sum_i^m {\alpha_i x^i},\ x^i \in C,\ \alpha_i \in (0,1],\ \sum{\alpha_i} = 1
> $$
> ​		$aff(x)$ 为 
> $$
> \sum{}\beta_i\sum{\alpha_i x^i} , \sum{\beta_i} =1, 
> $$
> ​	即是$C$中向量正组合，$aff(x) \subseteq aff(C)$ 反过来一样, $aff(C) = aff(conv(C))$可以证明；
>
> （2）证明 $aff(C) = aff(cl(C))$
>
> ​		a. 若$C$是闭集，则$C = cl(C)$，相应仿射包也相同
>
> ​		b. 若$C$是开集，$cl(C) = \{x| B(x,\epsilon) \cap C \neq \emptyset\}$ ,则 
> $$
> aff(x) = \sum{\beta_i x^i},\ x^i \in cl(C)
> $$
> ​		则一定能找到$\exists y^i \in C$, 使得$x^i = \epsilon + y^i$, $aff(x)$ 仍可以表示为$C$中向量的正组合, 两个仿射包相同；

## Cone 

**definition:**

- $C \subseteq \mathbb{R^n}$,  $x \in C$,  $\forall \alpha>0$ , if $\alpha x \in C$, 则$C$为一个**锥**，若$C$为凸集，$C$是凸锥;
  - 锥不一定是凸集；(如两个交叉直线)
  - 如果$C$是锥，一定有 $0 \in cl(C)$
- $C$中所有元素的**非负组合**的全体成为$C$的**生成锥**，记做$cone(C)$
  - 生成锥为凸锥且一定包含原点；
  - 不一定为闭集
  - $C$有限，其生成锥一定是闭集

**properties** of 锥集

- $C1 \cap C2$ 也是锥
- $C1 \times C2$ 笛卡尔积也是锥；
- $C1 + C2$ 向量和也是锥；
- $cl(C)$也是锥
- 线性变换$f(C)$也是锥

**properties** of $cone(C)$

- $cone(C) = cone(conv(C))$
- $aff(conv(C)) \subseteq aff(cone(C))$
- 若$0 \in conv(C)$, 则$aff(conv(C)) = aff(cone(C))$

**Caratheodory定理**

设 $C \subseteq \mathbb{R^n}$非空，则

- $cone(C)$中每一个向量均可以表示为$C$中$m$个线性独立向量的正组合；
- $conv(C)$ 中每一个不属于$C$的向量均可以表示为$C$中$m$个向量的凸组合，且 $m \leq n+ 1$;

乍一看这个定理貌似与凸包的定义相似，貌似是很显然的事情；区别在于$m \leq n+1$的证明上，因为从凸包的定义出发只能证明凸包中的向量可以由其他有限个向量的凸组合。而上述定理说明了不属于$C$的向量最多用$n+1$个向量表示出来；

> Proof:
>
> 令 $x \in C$,  $C \subseteq \mathbb{R^n}$ 则定义 $Y = \{(x, 1)\}$, $Y \subseteq \mathbb{R^{n+1}}$,
>
> 令 $\bar{x} \in conv(C) / C$, 由凸包定义， $ \bar{x}= \sum_i^I\alpha_ix^i\ ,\ x^i \in C,\ \sum{\alpha_i} = 1,\ \alpha_i \in (0,1]$
>
> 那么 $(\bar{x}, 1) = (\sum_i^I\alpha_ix^i, \sum_i^I\alpha_i) = \sum_i^I\alpha_i(x^i, 1) \subseteq cone(Y)$
>
> 则，依据定理第一条（证明略）：
>
> $(\bar{x}, 1) = \sum_i^M r_i(x^i, 1)$, 且 $M \leq n+1$

**Fig1**显示了第二条定理：

<img src="https://pic.superbed.cn/item/5dcea69a8e0e2e3ee94a517c.jpg" style="zoom:38%;" />

## 相对内部

相对内部的概念对之后凸分析有帮助，这里先给出定于与相关性质‘

**definition:** 

- 令集合$C$是非空凸集，若$x \in C$, $B(x, \epsilon)$是开球，若满足 $B\ \cap\ aff(C) \subseteq C$,则称$x$是$C$的**相对内部点**

- 全体相对内部点称作$C$的相对内部，记做$ri(C)$; 约定单点集的相对内部是他本身。

- 若$ri(C) = C$, 则称集合$C$是相对开的。$cl(C) / ri(C)$称作$C$的相对边界

**example**

$\mathbb{R^3}$中的集合$C=\left\{\boldsymbol{x} \in \mathbb{R}^{3} | x_{1}^{2}+x_{2}^{2} \leq 1, x_{3}=1\right\}$

仿射包与相对内部分别为：

$\operatorname{aff}(C)=\left\{\boldsymbol{x} \in \mathbb{R}^{3} | x_{3}=1\right\}, \operatorname{ri}(C)=\left\{\boldsymbol{x} \in \mathbb{R}^{3} | x_{1}^{2}+x_{2}^{2}<1, x_{3}=1\right\}$

<img src="https://pic.superbed.cn/item/5dceb2d88e0e2e3ee94c48ab.jpg" style="zoom:50%;" />

其内部是空集，闭包是非空凸集；可以看出三维空间中平面上的内部的定义就失效了，因为内点是相对$\mathbb{R^3}$定义的，而相对内点是针对仿射包定义的；

**Theorem：线段原理、延伸引理**

令集合$C$为非空凸集，

- 若$x \in ri(C)$,  $\bar{x} \in cl(C)$, 则连接$x$与$\bar{x}$的线段上，除$\bar{x}$外，所有点均属于$ri(C)$;
- $x \in ri(C)$ iif $\forall \bar{x} \in C$, 存在$\gamma > 0$似的 $x+\gamma(x-\bar{x}) \in C$

<img src="https://pic.superbed.cn/item/5dceb4bf8e0e2e3ee94c79dc.jpg" style="zoom:50%;" />

**properties** 

设集合$C$是非空凸集，则

- $ri(C)$是非空凸集；
- $\operatorname{aff}(\operatorname{ri}(C))=\operatorname{aff}(C), \operatorname{cl}(C)=\operatorname{cl}(\operatorname{ri}(C), \operatorname{ri}(C)=\operatorname{ri}(\operatorname{cl}(C))$

设$\bar{C}$是另一个非空凸集，则下面三个条件是等价的

- $ri(C) = ri(\bar{C})$
- $cl(C) = cl(\bar{C})$
- $ri(C) \subseteq \bar{C} \subseteq cl(C)$

## Reference

- 老师的课件
- [Stanford CVPX ~Boyd 第一讲的slides](http://web.stanford.edu/class/ee364a/lectures/sets.pdf)
- [泛函的内部定义](https://zhuanlan.zhihu.com/p/86233206)
- [Carathéodory's theorem (convex hull)]([https://en.wikipedia.org/wiki/Carath%C3%A9odory%27s_theorem_(convex_hull)](https://en.wikipedia.org/wiki/Carathéodory's_theorem_(convex_hull)))

