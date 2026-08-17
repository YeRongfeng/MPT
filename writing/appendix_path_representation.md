# 边界对齐随机路径坐标：附录推导草稿

> 本文件保存 `method_draft.md` 第 3.4 节移出的数学细节。它是附录候选，不直接
> 进入当前方法正文；符号将在论文 LaTeX 版本中统一。

## A. 曲线空间均值

记 $B=\{0,1,n-1,n\}$ 和 $F=\{2,\ldots,n-2\}$ 分别为边界与自由
控制点索引，$Q_i=(i/n,0)$ 为规范坐标中的直线控制多边形。B 样条积分平方
距离对应的 Gram 矩阵为

$$
\Gamma_{ij}
=\int_0^1N_{i,k}(\tau)N_{j,k}(\tau)\,\mathrm d\tau.
$$

给定边界控制点 $C_B^b$，在曲线度量下离直线基准最近的自由块为

$$
\mu_{b,F}
=Q_F-\Gamma_{FF}^{-1}\Gamma_{FB}(C_B^b-Q_B),
\qquad
\mu_{b,B}=C_B^b.
$$

该式是正文中约束二次最小化问题的分块闭式解。

## B. 相关协方差与白化

令基础单轴相关控制过程的协方差为

$$
(K_0)_{ij}
=\frac{\min(i,j)-ij/n}{n^2},
\qquad i,j\in\{0,\ldots,n\}.
$$

定义线性映射 $T$：

$$
T_{FF}=I,\qquad
T_{FB}=\Gamma_{FF}^{-1}\Gamma_{FB},\qquad
T_{B,:}=0.
$$

实现中使用的自由块协方差及其 Cholesky 因子为

$$
K_F=(TK_0T^\top)_{FF},\qquad K_F=LL^\top.
$$

两个平面坐标共享 $K_F$ 并彼此独立。令
$Y=\operatorname{mat}_{|F|\times2}(y)$，则

$$
y\sim\mathcal N(0,I_{d_y}),\qquad
C_F(y)=\mu_{b,F}+LY,\qquad
C_B(y)=C_B^b.
$$

这一写法与 `boundary_constrained_path.py` 中的 `free_from_fixed`、
`source_factor` 和 `canonical_source_mean` 一一对应。它不应被进一步概括为
整个随机分布的 $M$-正交投影，除非后续修改构造并补充相应证明。
