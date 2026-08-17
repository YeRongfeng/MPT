# 方法

## 1. 问题表述与信息边界

本文研究起伏、非结构化地形中面向机器人部署的全局路径学习。部署时，规划器依据
部分地形观测生成满足指定起终点位姿的二维全局几何参考路径，并使路线适应车辆的
空间支撑、倾覆稳定性与转向能力。该任务涉及三类作用方式不同的信息：当前观测仅能
刻画窗口内已有证据的规划支撑，起终点几何由任务准确给定，完整窗口地形及其稳定性
信息则只在训练阶段可用。无人机通过前沿探索与地形建图扩展车载感知范围，但在本文
系统中仅提供上游观测；路径跟踪与短时环境响应由下游局部规划器处理。

本文规划器采用固定物理范围 $\Omega$ 及其固定栅格作为模型接口。渐进式建图过程中，
输入张量的空间范围保持不变，窗口内具有地形证据且允许车辆进入的可靠规划支撑则随
观测推进而变化。为显式表示这一变化，令
$m:\Omega\to\{0,1\}$ 为 planning-support mask，并将可靠规划支撑记为
$\Omega_m=\{\mathbf r\in\Omega\mid m(\mathbf r)=1\}$。其中，$m(\mathbf r)=1$ 表示位置 $\mathbf r$ 已被观测且允许车辆中心进入，
$m(\mathbf r)=0$ 表示该位置不能作为可靠规划支撑；本文采用单通道支撑表述，将未观测
区域与已知禁入区域统一编码为零。该变量使生成器能够区分有效地形证据与法向填充值，
并将变化的规划支撑作为部署条件。其作用是界定可被模型使用的地形支撑，而不是恢复
未观测地形或独立提供避障保证。

部署条件写为

$$
c_{\mathrm{obs}}=(X_{\mathrm{obs}},S,G),\qquad
X_{\mathrm{obs}}=[\tilde n_x,\tilde n_y,\tilde n_z,m],
$$

其中 $S=(x_s,y_s,\psi_s)$ 和 $G=(x_g,y_g,\psi_g)$ 分别为起终点位姿，
$\tilde{\mathbf n}$ 为经 mask 处理的表面法向；在 $m=0$ 处，法向通道由约定噪声
替代，额外的 mask 通道保留其支撑语义。完整窗口地形记为 $X_{\mathrm{full}}$，作为
训练期特权信息参与地形可行性目标与离线审计，但不构成部署输入。

规划器输出几何路径 $p:[0,1]\to\mathbb R^2$。本文考虑起终点位置不重合的任务，
令 $d=\lVert G_{xy}-S_{xy}\rVert_2>0$，并令
$e(\psi)=(\cos\psi,\sin\psi)^\top$ 与 $b=(S,G)$。在归一化几何相位上定义边界路径空间

$$
\mathcal P_b^{(1)}
=
\left\{p\ \middle|\
\begin{aligned}
p(0)&=S_{xy}, & p(1)&=G_{xy},\\
p'(0)&=d\,e(\psi_s), & p'(1)&=d\,e(\psi_g)
\end{aligned}
\right\}.
$$

这里的导数关于归一化几何相位而非时间，$d$ 仅统一不同任务的端点导数尺度，不表示
车辆速度。由此，方法沿信息可用性形成一条完整链路：边界对齐表示将已知任务几何写入
生成坐标，条件生成器依据当前规划支撑预测未决的内部路线，expert route-prior learning
（ERPL）建立专家派生的条件路线先验，privileged multi-objective terrain adaptation
（PMTA）再从所得生成器出发，将训练期可用的特权物理监督吸收到同一
单步部署映射中。

## 2. Boundary-aligned trajectory representation

起终点位姿是任务已经确定的机器人几何，而学习变量应只描述尚未确定的内部路线。我们
因此构造 boundary-aligned generative coordinates：先统一路径的几何相位与任务坐标，
再通过固定边界控制点将端点位置和方向嵌入解码器。这样可减少采样密度、坐标系与曲线
参数差异在学习目标中引入的非物理变化。设示范位置曲线
$\gamma^\star:[0,1]\to\mathbb R^2$ 满足
$\gamma^\star(0)=S_{xy}$、$\gamma^\star(1)=G_{xy}$，并采用归一化弧长参数 $s$，
即其总弧长 $L>0$ 且 $\lVert \gamma^{\star\prime}(s)\rVert_2=L$ 几乎处处成立。
对于弦长 $d>0$，令 $\alpha=d/L$，并将 quintic smoothstep 记为 $h_5$；示范相位映射
与规范变换共同写为

$$
\begin{aligned}
h_5(\tau)&=10\tau^3-15\tau^4+6\tau^5,\\
\varphi(\tau)&=\alpha\tau+(1-\alpha)h_5(\tau),\\
p^\star(\tau)&=\gamma^\star(\varphi(\tau)),\\
\vartheta&=\operatorname{atan2}(y_g-y_s,x_g-x_s),\\
R_\phi&=\begin{bmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{bmatrix},\\
\hat p^\star(\tau)&=\frac{1}{d}R_{-\vartheta}\bigl(p^\star(\tau)-S_{xy}\bigr).
\end{aligned}
$$

由于 $0<\alpha\le 1$，该映射严格单调，并满足
$\varphi'(0)=\varphi'(1)=d/L$；归一化弧长参数化与该严格单调相位映射仅对示范曲线
重新参数化，因而保持其在物理任务坐标系中的几何像与点序。保向相似变换
$x\mapsto d^{-1}R_{-\vartheta}(x-S_{xy})$ 将起点、弦方向和弦长分别规范为
$(0,0)$、横轴正向和 $1$，并在平移、旋转和统一尺度意义下保持路径形状。

规范位置曲线因而具有统一端点 $(0,0)$ 和 $(1,0)$。$\hat p^\star$ 仅作为内部控制点的
位置拟合目标，其端点切向不被假定与任务 yaw 一致；任务方向分别变换为
$u_s=e(\psi_s-\vartheta)$ 与 $u_g=e(\psi_g-\vartheta)$，并由固定边界控制点单独写入。

为了把边界几何落实为生成空间的解析结构，规范路径采用 clamped cubic B-spline
表示。记 knot vector 为
$U=(u_0,\ldots,u_{n+4})$，其中
$u_0=u_1=u_2=u_3=0$，$0<u_4<\cdots<u_n<1$，且
$u_{n+1}=u_{n+2}=u_{n+3}=u_{n+4}=1$。规范路径写为

$$
\hat p(\tau)=\sum_{i=0}^{n}N^{U}_{i,3}(\tau)\hat C_i.
$$

在上述节点约定下，端点导数恒等式、端点导数系数以及四个固定边界控制点为

$$
\begin{aligned}
\hat p'(0)&=\beta_s(\hat C_1-\hat C_0),&\hat p'(1)=\beta_g(\hat C_n-\hat C_{n-1}),\\
\beta_s&=\frac{3}{u_4-u_3},&
\beta_g=\frac{3}{u_{n+1}-u_n},\\
\hat C_0^b&=(0,0),&
\hat C_1^b=\hat C_0^b+\frac{u_s}{\beta_s},\\
\hat C_n^b&=(1,0),&
\hat C_{n-1}^b=\hat C_n^b-\frac{u_g}{\beta_g}.
\end{aligned}
$$

对于均匀 clamped knot vector，$\beta_s$ 与 $\beta_g$ 取同一常数。由上述恒等式，
起终点位置及方向可直接写入四个边界控制点。其余控制点仅描述两端之间的内部形变，
因此网络只预测内部路径自由度。令 $F=\{2,\ldots,n-2\}$ 为自由索引，
$B=(0,1,n-1,n)$ 为边界索引的固定顺序，$d_y=2|F|=2(n-3)$；按该顺序记
$\hat C_B^b=(\hat C_0^b,\hat C_1^b,\hat C_{n-1}^b,\hat C_n^b)$。

边界控制点确定后，我们以一组固定、边界一致的仿射坐标描述相关的内部形变。对
$y\in\mathbb R^{d_y}$，令 $Y(y)\in\mathbb R^{|F|\times2}$ 按自由索引逐行满足
$\operatorname{vec}_F(Y(y))=(Y_{1,x},Y_{1,y},\ldots,Y_{|F|,x},Y_{|F|,y})^\top=y$。
自由控制点及完整控制点序列定义为

$$
\begin{aligned}
\hat C_F(y)&=\hat\mu_{b,F}+L_FY(y),
\qquad L_FL_F^\top=K_F^{\mathrm{br}},\\
\hat C_i(y)&=
\begin{cases}
\hat C_i^b, & i\in B,\\
\hat C_{F,i}(y), & i\in F,
\end{cases}
\qquad i=0,\ldots,n .
\end{aligned}
$$

其中 $\hat C_{F,i}(y)$ 表示 $\hat C_F(y)$ 中与自由索引 $i$ 对应的二维控制点。
$\hat\mu_{b,F}$ 是边界控制点相对直线弦发生变化时的 $L_2$ boundary compensation，
$K_F^{\mathrm{br}}$ 是与该补偿一致的 boundary-projected Brownian-bridge covariance，
$L_F$ 为其 Cholesky 因子，两个空间坐标共享该因子。边界补偿均值
$\hat\mu_{b,F}$、协方差 $K_F^{\mathrm{br}}$ 及其 Cholesky 因子 $L_F$ 的具体构造见
附录 A。该仿射映射完整规定了边界相关均值与固定相关形变，
不引入可学习参数。

将这些按 $i=0,\ldots,n$ 排列的规范控制点插值成规范路径后，直接映射回任务坐标系：

$$
D_b(y)(\tau)
=S_{xy}+d\,R_{\vartheta}\sum_{i=0}^{n}N^{U}_{i,3}(\tau)\hat C_i(y).
$$

$D_b$ 将白化自由坐标依次映射为控制点、规范 B-spline 和任务坐标中的物理路径。由
边界控制点的解析构造，对任意 $y$ 均有 $D_b(y)\in\mathcal P_b^{(1)}$。因此，
起终点位置及两端一阶方向由生成坐标解析满足，网络只需学习内部路径自由度。

示范曲线经相同规范化后也编码到该自由坐标空间。固定解析边界控制点，仅拟合自由控制
块，并通过 $L_F$ 的线性求解得到白化坐标：

$$
\begin{aligned}
\hat p_{Z_F}(\tau)
&=\sum_{i\in B}N^U_{i,3}(\tau)\hat C_i^b+\sum_{i\in F}N^U_{i,3}(\tau)Z_i,\\
\hat C_F^\star
&\in\arg\min_{Z_F\in\mathbb R^{|F|\times2}}
\int_0^1\left\|\hat p^\star(\tau)-\hat p_{Z_F}(\tau)\right\|_2^2\,\mathrm d\tau,\\
L_FW^\star&=\hat C_F^\star-\hat\mu_{b,F},\qquad
y^\star=\operatorname{vec}_F(W^\star).
\end{aligned}
$$

其中 $Z_i$ 表示候选自由控制块 $Z_F$ 中与索引 $i$ 对应的二维行向量，
$\operatorname{vec}_F$ 采用与 $Y(y)$ 相同的逐自由索引、逐 $(x,y)$ 坐标顺序。

在这一自由坐标空间中，定义 Gaussian source 及其诱导的物理路径分布为

$$
\begin{aligned}
q_0(y)&=\mathcal N(0,I_{d_y}),\\
\operatorname{vec}_F(\hat C_F)\mid b
&\sim\mathcal N\!\left(
\operatorname{vec}_F(\hat\mu_{b,F}),K_F^{\mathrm{br}}\otimes I_2
\right),\\
p_0^b&=(D_b)_\#q_0.
\end{aligned}
$$

随机形变由此仅作用于内部控制点，边界块始终由任务位姿确定。上述拟合与白化过程使
source、示范目标和生成状态共享同一自由坐标空间。
我们将 $p_0^b$ 称为 Gaussian trajectory prior：它是标准高斯源经边界对齐自由坐标与
$D_b$ 诱导的边界一致路径分布。该 source prior 规定生成过程的结构化起点，不同于
ERPL 通过专家路径监督获得的 conditional route prior，后者刻画给定
条件下的长程路线偏好。至此，轨迹表示提供了解析边界性质和结构化随机源；规划支撑、稳定性与曲率
可行性仍由后续学习目标塑造并通过实验检验。

## 3. Planning-support-conditioned generator

边界表示消除已知任务几何后，剩余的学习问题是：固定窗口内变化的可靠支撑与地形证据
如何共同决定内部路线。规划支撑 mask 与法向场为此构成互补条件；$m=1$ 的区域同时
具有地形观测并允许车辆中心进入，$m=0$ 的区域则不能作为可靠规划支撑。在本文的单通道
表述中，未知与已知禁入两种原因共享零值语义。法向场描述支撑内的地形几何，mask 则
防止填充值被解释为有效观测，并使同一生成器能够响应不同的支撑范围和局部阻塞。

当数据不提供现成 mask 时，训练支撑在车辆配置空间中构造：候选区域经车辆半径腐蚀，
并通过起终点及其连通性检查。ERPL 的训练样本要求专家路径保留在合成支撑内；PMTA
使用的训练条件则允许支撑变化在示范路线中段形成局部阻塞，使物理适配面对偏离原示范
的规划条件。具体采样分布和拒绝参数在 Experimental Setup 中给出。由此得到的 $m$
已是车辆中心可进入的配置空间支撑，后续物理代价不再对其重复腐蚀。

上游点云首先被栅格化为 2.5D 高程地图。参照 Capsizing-Aware Planner
（CAP; Zhang et al., 2025），局部支撑平面拟合与地表回归产生连续高程-法向场
$\mathcal F_X(\mathbf r)=[z_X(\mathbf r),\mathbf n_X(\mathbf r)]$。

本文沿用 CAP 的地形回归、点云预处理与车辆支撑几何：在其 stability-pyramid 模型中，
若重力作用线在地面支撑平面上的投影落入车辆接触点构成的稳定多边形，则该位置-朝向
满足非倾覆判据。由此继承位置 $\mathbf r$ 处的可通行朝向集合
$\Theta_X(\mathbf r)$；本文进一步将该集合转化为可微的路径级稳定性余度。相应的不稳定集合为
$\mathcal U_X=\{(\mathbf r,\psi)\mid \psi\notin\Theta_X(\mathbf r)\}$。

为统一位置与朝向差异，令
$\operatorname{wrap}(\Delta\psi)\in[-\pi,\pi)$ 表示周期角差，并以
$w_\psi>0$（yaw weight）将角度差换算为等效空间距离。乘积空间
$\Omega\times\mathbb S^1$ 上采用 yaw-periodic weighted metric；其中
$w_\psi$ 的单位为长度/弧度，其取值在 Experimental Setup 中给出。对
$\zeta=(\mathbf r,\psi)$，记
$\operatorname{dist}_{w_\psi}(\zeta,\mathcal A)=\inf_{a\in\mathcal A}\mathfrak d_{w_\psi}(\zeta,a)$，
则 stability signed distance 定义为

$$
\begin{aligned}
\mathfrak d_{w_\psi}\!\left((\mathbf r,\psi),(\mathbf r',\psi')\right)
&=\sqrt{\lVert\mathbf r-\mathbf r'\rVert_2^2
+w_\psi^2\operatorname{wrap}(\psi-\psi')^2},\\
d_X^{\mathrm{stab}}(\zeta)
&=\operatorname{dist}_{w_\psi}(\zeta,\mathcal U_X)
-\operatorname{dist}_{w_\psi}(\zeta,\mathcal U_X^c).
\end{aligned}
$$

其中稳定侧为正，不稳定侧为负。该连续定义给出理想的 yaw-periodic stability metric；
计算时，在位置栅格与有限个均匀 yaw bins 上离散 $\mathcal U_X$，通过 yaw 维周期平铺
构造 signed-distance grid，并以空间--yaw 三线性插值连续读取余度。栅格分辨率与 yaw
bin 数量在 Experimental Setup 中给出。对 $N\ge4$ 个均匀路径采样相位
$\tau_i=(i-1)/(N-1)$，$i=1,\ldots,N$，B-spline 的解析一阶导数确定车辆 yaw 及其
stability margin：

$$
\psi_i=\operatorname{atan2}\!\left(p_y'(\tau_i),p_x'(\tau_i)\right),\qquad
s_i=s_X(p;\tau_i)
=d_X^{\mathrm{stab}}\!\left(p(\tau_i),\psi_i\right).
$$

路径 yaw 的前向值始终采用精确的 $\operatorname{atan2}$；仅在反向传播时，以由最小
线段长度导出的正下限截断其分母，从而限制近零切向量处的梯度而不改变前向朝向。由此
得到的插值 stability margin 将在 PMTA 中转化为路径级物理监督。

对于给定边界条件 $b$，生成器定义为

$$
\begin{aligned}
f_\theta(c_{\mathrm{obs}},z_t,t,r)
&\longmapsto \hat y_0\in\mathbb R^{d_y},\\
q_\theta(y\mid c_{\mathrm{obs}}),\qquad
p_\theta(\cdot\mid c_{\mathrm{obs}})
&=(D_b)_\#q_\theta(\cdot\mid c_{\mathrm{obs}}).
\end{aligned}
$$

其中 $z_t$ 为自由路径空间中的生成状态，$t$ 与 $r$ 为 MeanFlow 时间条件，
$\hat y_0$ 为预测的数据端坐标。作为 $f_\theta$ 的具体实现，四通道
$X_{\mathrm{obs}}$ 经卷积地形编码器转化为空间 map tokens，$z_t$ 按 $|F|$ 个二维自由
坐标组织为 path/query tokens。归一化起终点位姿与时间嵌入 $t$、$t-r$ 融合为全局条件，
用于调制 DiT-style 路径主干；任务--时间条件化的进度 queries 先以交叉注意读取 map
tokens 形成路径对齐的 guidance tokens，路径主干再以交叉注意读取这些地形引导。最终，
逐 path-token 预测头输出二维数据端自由坐标，并拼接为 $\hat y_0$。该结构用于实现上述
条件映射，其层数、宽度与注意力头数在 Experimental Setup 中给出。

训练将在不同 $(t,r)$ 上约束该条件分布，而部署仅调用其单步 endpoint。对
$\xi\sim\mathcal N(0,I_{d_y})$，endpoint 样本由
$D_b(f_\theta(c_{\mathrm{obs}},\xi,1,0))$ 得到；第四节在 PMTA 之后给出最终部署调用。
该条件映射把变化的可靠规划支撑转化为内部路线选择，解析解码器则维持任务边界；路线
的地形物理性质由下述 ERPL 与特权地形适配塑造，并由实验评价。

## 4. Expert route-prior learning, privileged multi-objective terrain adaptation, and deployment

全局路线生成需要两类互补知识：示范路径提供长程绕行结构，可微物理代价则揭示规划
支撑、倾覆稳定性和曲率违反，其中稳定性监督依赖仅在训练期可用的完整地形。ERPL
从专家路径建立具有长程路线语义的条件生成器；随后，PMTA 从该生成器初始化，
在与部署一致的 endpoint 上引入物理监督，而不改变部署输入或增加测试时优化。两种学习
功能始终只以 $c_{\mathrm{obs}}$ 与 source 作为生成器输入；完整地形仅在 PMTA 中用于
稳定性目标和离线审计。该顺序设计相对联合训练的作用由消融实验检验。

**Expert route-prior learning (ERPL).** 给定条件
$c_{\mathrm{obs}}$、示范自由坐标 $y^\star$ 和高斯 source
$\xi\sim\mathcal N(0,I_{d_y})$，Path MeanFlow 在二者之间采用线性概率路径：

$$
z_t=(1-t)y^\star+t\xi,\qquad
u=\xi-y^\star,\qquad
0\le r\le t\le1.
$$

由生成器的数据端预测定义平均速度参数化，其中 $\epsilon>0$ 为数值稳定项：

$$
v_\theta(c_{\mathrm{obs}},z_t,t,r)
=\frac{z_t-f_\theta(c_{\mathrm{obs}},z_t,t,r)}{t+\epsilon}.
$$

为覆盖有序时间区间，时间采样分布 $\pi_{tr}$ 混合连续 ordered component 与 endpoint
atom。具体地，先抽取 $a_1,a_2\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(\mu_t,\sigma_t^2)$，
将二者分别经 sigmoid 后裁剪到预设内区间得到 $t_1,t_2$，再排序为
$t=\max(t_1,t_2)$、$r=\min(t_1,t_2)$；随后以概率 $p_{\mathrm{end}}$ 将该时间对替换为
部署端点 $(t,r)=(1,0)$，否则保留满足 $0<r\le t<1$ 的 ordered pair。
$p_{\mathrm{end}}$ 控制 endpoint 时间对在训练批次中的出现频率；它与
$\lambda_{\mathrm{end}}$ 的监督权重作用不同。$\mu_t$、$\sigma_t$、裁剪边界以及
$p_{\mathrm{end}}$ 的具体取值在 Experimental Setup 中给出。在固定条件
$c_{\mathrm{obs}}$ 下，令

$$
\begin{aligned}
\dot v_\theta
&=D_{(u,1,0)}v_\theta(c_{\mathrm{obs}},z_t,t,r),\\
\bar v_\theta
&=v_\theta+(t-r)\,\operatorname{sg}\!\left(
\operatorname{clip}(\dot v_\theta,-\kappa_v,\kappa_v)\right),
\end{aligned}
$$

其中 $\dot v_\theta$ 表示对 $(z_t,t,r)$ 沿方向 $(u,1,0)$ 的方向导数，
$\operatorname{sg}$ 表示停止梯度，$\kappa_v$ 的具体取值在 Experimental Setup
中给出。取
$\lambda_{\mathrm{end}}=0.25$，成对的 Path MeanFlow 目标定义为

$$
\begin{aligned}
\mathcal J_{\mathrm{PMF}}(\theta;c_{\mathrm{obs}},\xi,y^\star,t,r)
= {}&
\operatorname{MSE}(\bar v_\theta,u)\\
&+\lambda_{\mathrm{end}}\,
\operatorname{MSE}\!\left(f_\theta(c_{\mathrm{obs}},\xi,1,0)-y^\star\right),
\end{aligned}
$$

其中 $\operatorname{MSE}$ 对自由坐标与批内样本取平均。第一项在随机 $(t,r)$ 上匹配
MeanFlow 速度，第二项直接监督实际部署的 endpoint。

在示范集合上优化

$$
\mathcal L_{\mathrm{ERPL}}(\theta)
=\mathbb E_{(c_{\mathrm{obs}},y^\star),\,\xi,\,(t,r)\sim\pi_{tr}}
\left[
\mathcal J_{\mathrm{PMF}}(\theta;c_{\mathrm{obs}},\xi,y^\star,t,r)
\right].
$$

$\mathcal L_{\mathrm{ERPL}}$ 将专家路径中的长程绕行结构与内部几何偏好迁移到条件生成器，
由此形成 expert-derived conditional route prior。该 learned route prior 是模仿目标定义的
条件路线分布，不同于第二节规定生成起点的 Gaussian trajectory prior。该模仿目标
并不直接优化完整地形实例上的物理可行性，这一监督缺口构成后续特权地形适配的出发点。

**Privileged multi-objective terrain adaptation (PMTA).**
接近专家示范并不等价于降低当前规划支撑、完整地形稳定性与车辆曲率约束共同定义的
物理风险。将这些异质风险预先压缩为固定加权和，会在整个训练过程中隐含一组全局
不变的风险交换率；当分量梯度发生冲突时，加权和的下降方向可能以其中一项风险的上升
换取另一项的下降。为此，PMTA
将三类路径级物理风险保留为独立目标，并在每次适配更新中协调其参数梯度。实际
训练域中是否存在梯度冲突与共同下降方向，由 Experiments 的 gradient-geometry
analysis 独立检验，而非作为普遍性质预设。

PMTA 从 ERPL 所得 route generator 出发。记其 checkpoint 参数为
$\theta_{\mathrm{ERPL}}$，初始化 $\theta_{\mathrm{PMTA}}\leftarrow\theta_{\mathrm{ERPL}}$，
下文 PMTA 目标与梯度推导中的演化参数统一记为
$\theta\equiv\theta_{\mathrm{PMTA}}$。该初始化使三项可微地形代价直接作用于与部署
一致的 endpoint
$(t,r)=(1,0)$。因此，特权监督改变的是部署映射的参数，而不是部署时可见的信息。

以下三项目标分别度量规划支撑违反、稳定性不足和曲率违反。令
$\rho>0$ 为地图分辨率，并沿用上述 $N$ 个稠密路径采样点；令
$b_{\mathrm{safe}}\in\mathbb Z_{\ge0}$ 为地图边界向内收缩的安全像素数；令
$\alpha_F>0$ 和 $\alpha_S>0$ 分别为无量纲 forbidden 与 stability softplus 尺度，
$d_{\mathrm{safe}}>0$ 为具有长度单位的稳定性余度，$\kappa_{\max}>0$ 为具有逆长度
单位的曲率阈值，$\ell_{\min}>0$ 为最小线段长度，$w_{\mathrm{short}}\ge0$ 为无量纲
短线段违反权重。
三项尾部比例满足 $\eta_F,\eta_S,\eta_\kappa\in(0,1]$，具体取值统一在
Experimental Setup 中给出。

为同时评估采样点与相邻采样点之间的支撑违反，令
$p_i=p(\tau_i)$、$i=1,\ldots,N$ 为稠密 B-spline 采样点，并在每条相邻线段上以
不大于 $\rho/2$ 的间距插入内部点集合 $\mathcal Q_{\mathrm{seg}}(p)$。三类代价的
采样集合定义为

$$
\mathcal Q_F(p)=\{p_i\}_{i=1}^{N}\cup\mathcal Q_{\mathrm{seg}}(p),
\qquad
\mathcal Q_S(p)=\mathcal Q_\kappa(p)=\{p_i\}_{i=1}^{N}.
$$

为使目标集中于路径上违反最严重的区域，对 $M$ 个逐点量
$a_1,\ldots,a_M$ 和 $\eta\in(0,1]$，令
$k=\max\{1,\lceil\eta M\rceil\}$，$I_k$ 为最大 $k$ 个量的索引，并定义
尾部均值

$$
\operatorname{TailMean}_{\eta}(a_1,\ldots,a_M)
=\frac{1}{k}\sum_{j\in I_k}a_j.
$$

并记 $[a]_+=\max(a,0)$。

**Planning-support violation.** 对 $q=(x,y)\in\mathcal Q_F(p)$，记
$\delta_{\mathrm{mask}}(q;m)$ 为直接由配置空间 planning-support mask $m$ 构造的
signed-distance field 的双线性采样值，可规划侧为正、禁入侧为负；由于 $m$ 已编码
车辆 footprint 支撑，此处不再执行额外腐蚀。进一步将物理地图
边界向内收缩 $b_{\mathrm{safe}}\rho$，令
$(x_{\min},x_{\max},y_{\min},y_{\max})$ 表示收缩后的物理范围，则边界 clearance
与复合 clearance 为

$$
\begin{aligned}
\delta_{\mathrm{box}}(q)
&=\min\{x-x_{\min},x_{\max}-x,y-y_{\min},y_{\max}-y\},\\
\delta_F(q;m)&=\min\bigl(\delta_{\mathrm{mask}}(q;m),\delta_{\mathrm{box}}(q)\bigr).
\end{aligned}
$$

该最小值保持可规划侧为正的符号约定，但不被解释为二者交集的精确欧氏 signed
distance。相应的 forbidden 代价为

$$
C_F(p;m)
=\operatorname{TailMean}_{\eta_F}
\left\{
\frac{\operatorname{softplus}\left(-\alpha_F\delta_F(q;m)/\rho\right)}{\alpha_F}
\;\middle|\;q\in\mathcal Q_F(p)
\right\}.
$$

**Stability violation.** 对每个 $p_i$，完整地形在路径切向 yaw $\psi_i$ 下给出
stability margin
$s_i=d_{X_{\mathrm{full}}}^{\mathrm{stab}}(p_i,\psi_i)$，由此定义

$$
C_S(p;X_{\mathrm{full}})
=\operatorname{TailMean}_{\eta_S}
\left\{
\frac{\operatorname{softplus}\left(\alpha_S(1-s_i/d_{\mathrm{safe}})\right)}
{\alpha_S}
\;\middle|\;i=1,\ldots,N
\right\}.
$$

**Curvature violation.** B-spline 的解析导数同时给出转弯与短线段违反量。记
$g_i=p'(\tau_i)$、$h_i=p''(\tau_i)$，并以
$s_{\mathrm{floor}}=\ell_{\min}(N-1)$ 稳定近零切向量处的归一化，则

$$
v_{\mathrm{turn},i}
=\frac{
\left[
\left|g_{i,x}h_{i,y}-g_{i,y}h_{i,x}\right|
-\kappa_{\max}\lVert g_i\rVert_2^3
\right]_+
}
{\kappa_{\max}\left(\lVert g_i\rVert_2^3+s_{\mathrm{floor}}^3\right)}.
$$

相邻采样点之间的短线段违反量定义为

$$
\begin{aligned}
r_{\mathrm{short},j}
&=\frac{\left[\ell_{\min}-\lVert p_{j+1}-p_j\rVert_2\right]_+}{\rho},
\qquad j=1,\ldots,N-1,\\
v_{\mathrm{short},1}&=r_{\mathrm{short},1},\qquad
v_{\mathrm{short},N}=r_{\mathrm{short},N-1},\\
v_{\mathrm{short},i}&=\max(r_{\mathrm{short},i-1},r_{\mathrm{short},i})
\quad (1<i<N).
\end{aligned}
$$

综合两类几何违反后，曲率代价为

$$
C_\kappa(p)
=\operatorname{TailMean}_{\eta_\kappa}
\left\{
\log\left(1+v_{\mathrm{turn},i}\right)
+w_{\mathrm{short}}v_{\mathrm{short},i}
\;\middle|\;i=1,\ldots,N
\right\}.
$$

三项连续代价共同定义 PMTA 的多目标物理适配问题。令 $\mathcal B$ 为一批训练条件
$\chi=(c_{\mathrm{obs}},X_{\mathrm{full}},m)$；对每个 $\chi$ 采样 $K\ge1$ 个独立
source，并在部署 endpoint 上生成路径：

$$
\begin{aligned}
\xi_{\chi k}&\sim\mathcal N(0,I_{d_y}),\\
\hat y_{\chi k}^\theta
&=f_\theta(c_{\mathrm{obs},\chi},\xi_{\chi k},1,0),\\
p_{\chi k}^\theta&=D_{b_\chi}(\hat y_{\chi k}^\theta).
\end{aligned}
$$

由此在当前训练批次与所采样 source 上形成一个向量平均：

$$
\begin{aligned}
\mathbf J_{\mathrm{PMTA}}(\theta)
&=
\begin{bmatrix}
\overline C_F(\theta)\\
\overline C_S(\theta)\\
\overline C_\kappa(\theta)
\end{bmatrix}\\
&=\frac{1}{|\mathcal B|K}
\sum_{\chi\in\mathcal B}\sum_{k=1}^{K}
\begin{bmatrix}
C_F(p_{\chi k}^\theta;m_\chi)\\
C_S(p_{\chi k}^\theta;X_{\mathrm{full},\chi})\\
C_\kappa(p_{\chi k}^\theta)
\end{bmatrix}.
\end{aligned}
$$

PMTA 实际优化该向量目标
$\mathbf J_{\mathrm{PMTA}}$。令 $g_a=\nabla_\theta\overline C_a(\theta)$，
$a\in\{F,S,\kappa\}$，并定义概率单纯形
$\Delta_3=\{\omega\in\mathbb R_{\ge0}^3\mid\omega_F+\omega_S+\omega_\kappa=1\}$。
为实例化这一梯度协调，PMTA 采用
multiple-gradient descent algorithm（MGDA；Désidéri, 2012），在该单纯形上求取
目标梯度凸包中的最小范数元素：

$$
\begin{aligned}
\omega^\star
&\in\arg\min_{\omega\in\Delta_3}
\left\|\omega_F g_F+\omega_S g_S+\omega_\kappa g_\kappa\right\|_2^2,\\
g^\star
&=\omega_F^\star g_F+\omega_S^\star g_S+\omega_\kappa^\star g_\kappa.
\end{aligned}
$$

$g^\star$ 是从当前分量梯度凸包中选出的最小范数元素，作为更新单步生成器的局部协调
梯度；$\omega^\star$ 仅刻画当前参数点上的凸组合选择，随后通过 PMTA 参数更新将三类
物理监督吸收到单步生成器。该局部求解不被解释为对全部训练迭代的共同下降、Pareto
最优性或收敛性保证。

给定学习率 $\eta_{\mathrm{sgd}}>0$ 与全局梯度阈值 $\tau_g>0$，定义
$\operatorname{Clip}_{\tau_g}(g)=g\tau_g/\max\{\tau_g,\lVert g\rVert_2\}$，并通过
plain-SGD 更新

$$
\theta_{\mathrm{PMTA}}\leftarrow
\theta_{\mathrm{PMTA}}-\eta_{\mathrm{sgd}}
\operatorname{Clip}_{\tau_g}(g^\star).
$$

$\eta_{\mathrm{sgd}}$ 与 $\tau_g$ 在 Experimental Setup 中给出。这里采用 plain SGD
和全局范数 clipping，是因为 clipping 只对 $g^\star$ 作统一缩放并保留其方向；
自适应逐坐标预条件通常不会保留这一方向。该更新不将连续违反量转化为硬约束；
相应 hard threshold 仅用于监控和独立评价。

这里的 privileged 指 $X_{\mathrm{full}}$ 及其 stability field 仅用于 PMTA 的稳定性
目标和离线审计，而生成器始终只接收 $c_{\mathrm{obs}}$ 与 source。ERPL
提供示范支撑的全局路线结构，PMTA 则在同一 endpoint generator 上协调三类物理风险，
并将包括完整地形稳定性信息在内的训练期监督吸收到参数 $\theta_{\mathrm{PMTA}}$ 中，
而不改变部署输入。由于适配目标仅由
$C_F$、$C_S$ 与 $C_\kappa$ 构成，且不含 demonstration anchor、distribution-matching
term 或 mode-preservation regularizer，适配后的生成器能否保留 ERPL 所建立的结构与
source-conditioned coverage 是需要独立实验检验的经验性质，而非该目标所保证的性质。

部署的基本单位为一次 replanning invocation。规划器采样一个标准高斯 source，执行
一次 endpoint 前向并完成解析解码：

$$
\begin{aligned}
\xi&\sim\mathcal N(0,I_{d_y}),\\
\hat y&=f_{\theta_{\mathrm{PMTA}}}(c_{\mathrm{obs}},\xi,1,0),\\
\hat p&=D_b(\hat y).
\end{aligned}
$$

由 $\hat p\in\mathcal P_b^{(1)}$，端点位置与一阶方向无需额外投影。一次调用只使用
部分地形观测、planning-support mask、起终点位姿和一个 source，并将所得全局参考
路径交由下游局部规划器；部署阶段不访问 $X_{\mathrm{full}}$，也不附加在线轨迹优化或
多路径选择。该部署契约保留了解析任务边界和一次前向推理，而地形可行性仍是训练塑形
后需要在未见环境与真实系统中验证的经验性质。
