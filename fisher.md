## 1) 公式（对应当前实现）

设第 $k$ 次参数更新时模型参数为 $\theta_k$，参考参数为 $\bar\theta_k$（上一轮更新后的参数快照），可训练参数索引为 $j$。

### (a) 安全损失
由重建轨迹 $\tau_\theta$ 计算：

$$
\mathcal L_{\text{safe}}(\theta)
= \texttt{cost\_on\_dense\_trajectory}\!\big(\tau_\theta,\text{map},\text{start},\text{goal}\big)
$$

若非有限值，则回退：

$$
\mathcal L_{\text{safe}} \leftarrow 0
$$

### (b) Fisher 对角近似（EMA）
先取安全损失对参数的梯度：

$$
g_j^{(k)}=\nabla_{\theta_j}\mathcal L_{\text{safe}}(\theta_k)
$$

用梯度平方做对角 Fisher 近似并 EMA 更新：

$$
F_{j}^{(k)}
=\beta\,F_{j}^{(k-1)} + (1-\beta)\,(g_j^{(k)})^2
$$

其中当前实现 $\beta=0.95$。

### (c) 近端惩罚项

$$
\mathcal L_{\text{prox}}(\theta_k)
= \frac{1}{J}\sum_{j=1}^{J}\operatorname{mean}\!\left(
F_j^{(k)}\odot(\theta_{k,j}-\bar\theta_{k,j})^2
\right)
$$

### (d) 第二阶段训练时的 capsize 损失

$$
\mathcal L_{\text{capsize}}(\theta_k)
=
\mathcal L_{\text{safe}}(\theta_k)
+\lambda_{\text{prox}}\mathcal L_{\text{prox}}(\theta_k)
$$

其中当前实现 $\lambda_{\text{prox}}=\texttt{proxy\_scale}=0.5$。

> 验证时（`is_training=False`）不加近端项，即  
> $$
> \mathcal L_{\text{capsize}}=\mathcal L_{\text{safe}}.
> $$

---

## 2) 伪代码（对应当前实现）

```text
Initialize:
    ref_params = None
    fisher_diag = None
    beta = 0.95
    proxy_scale = 0.5

For each batch in stage-2:

    # 1) 轨迹构造
    if training:
        x0_pred = z_t - t * u_out
        traj = BSpline(start, x0_pred, goal)
    else:
        traj = sample_chain(model, map, start, goal)

    # 2) 基础安全损失
    safe_loss = cost_on_dense_trajectory(traj, map_info, cost_map, start, goal)
    if not finite(safe_loss):
        safe_loss = 0

    capsize_loss = safe_loss

    # 3) 训练时加 Fisher 近端
    if training:
        trainable_params = params(requires_grad=True)

        if ref_params is None or shape changed:
            ref_params = clone(trainable_params)
            fisher_diag = ones_like(trainable_params)

        grads = grad(safe_loss, trainable_params, retain_graph=True)

        for each param j:
            g2 = 0 if grads[j] is None else grads[j]^2
            fisher_diag[j] = beta * fisher_diag[j] + (1 - beta) * g2

        fisher_penalty = mean_j( mean( fisher_diag[j] * (param[j] - ref_params[j])^2 ) )

        capsize_loss = safe_loss + proxy_scale * fisher_penalty

    # 4) 总损失中使用
    total_loss = w_main * main_loss + w_tangent * tangent_loss + w_capsize * capsize_loss

    backward(total_loss)
    optimizer.step()

    # 5) 每步后刷新近端锚点
    if stage == 2:
        ref_params = clone(current_trainable_params)