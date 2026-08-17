# Appendix A. Projected Brownian-bridge covariance construction

本附录给出主文中 boundary-aligned Gaussian trajectory prior 使用的
boundary-projected Brownian-bridge covariance。沿用主文的 clamped cubic
B-spline knot vector $U$、自由索引 $F=\{2,\ldots,n-2\}$、边界索引顺序
$B=(0,1,n-1,n)$ 以及边界控制块
$\hat C_B^b=(\hat C_0^b,\hat C_1^b,\hat C_{n-1}^b,\hat C_n^b)$。令直线弦控制点为
$\bar C_i=(i/n,0)$，并以 $\bar C_F$ 与 $\bar C_B$ 分别表示其自由块与边界块。
B-spline 基函数的曲线级 $L_2$ 内积、边界补偿矩阵、补偿均值与离散 Brownian-bridge
协方差为

$$
\begin{aligned}
H_{ij}
&=\int_0^1N^U_{i,3}(\tau)N^U_{j,3}(\tau)\,\mathrm d\tau,
\qquad i,j=0,\ldots,n,\\
A&=H_{FF}^{-1}H_{FB},\\
\hat\mu_{b,F}&=\bar C_F-A(\hat C_B^b-\bar C_B),\\
\Sigma^{\mathrm{br}}_{ij}
&=\frac{\min(i,j)-ij/n}{n^2},
\qquad i,j=0,\ldots,n .
\end{aligned}
$$

投影矩阵 $P$ 仅有两个非零分块 $P_{FF}=I$ 与 $P_{FB}=A$，其余分块为零。于是自由块
上的 boundary-projected Brownian-bridge covariance 及其 Cholesky 因子为

$$
K_F^{\mathrm{br}}\equiv K_F=[P\Sigma^{\mathrm{br}}P^\top]_{FF},
\qquad L_FL_F^\top=K_F^{\mathrm{br}}.
$$

其中 $\hat\mu_{b,F}$ 是边界控制点相对直线弦发生变化时，使整条 B-spline 曲线位置扰动
最小的 $L_2$ 补偿；$L_F$ 是固定的、与边界补偿一致的平滑轨迹协方差的 Cholesky 因子，
两个空间坐标共享该因子。
