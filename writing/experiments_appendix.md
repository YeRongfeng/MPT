# 实验附录（实现参数草稿）

## A. MeanFlow 与优化设置

ERPL 使用 [优化器]，学习率为 [数值]，global gradient clipping 为 [数值]。MeanFlow 的 velocity coefficient 为 $\kappa_v=$ [数值]，时间条件采用 [采样分布与截断]，endpoint atom probability 为 $p_{\mathrm{end}}=$ [数值]。PMTA 使用 plain SGD，学习率为 [数值]，global gradient clipping 为 [数值]。

## B. PMTA 几何与目标尺度

PMTA 的几何参数为 $b_{\mathrm{safe}}=$ [数值] px、$d_{\mathrm{safe}}=$ [数值] m、$\ell_{\min}=$ [数值] m 和 $w_{\mathrm{short}}=$ [数值]。目标 reduction 参数为 $(\eta_F,\eta_S,\eta_\kappa)=$ [数值] 与 $(\alpha_F,\alpha_S)=$ [数值]；$C_F$、$C_S$ 和 $C_\kappa$ 的具体尺度定义及 gradient normalization 采用 [具体定义/不采用]。
