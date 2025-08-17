import torch
import math
from typing import Tuple, List, Optional, Sequence, Union

# ------------------ 公共辅助函数（周期化处理） ------------------
def normalize_angle(a: torch.Tensor) -> torch.Tensor:
    """
    把角度张量归一到 [-pi, pi)
    支持标量 tensor 或向量 tensor
    """
    TWO_PI = 2.0 * math.pi
    return (torch.remainder(a + math.pi, TWO_PI) - math.pi)

def angular_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    返回 |normalize(a - b)| 最小角差（弧度，>=0）
    支持广播：a, b 可以不同 shape，返回广播后的绝对角差
    """
    d = normalize_angle(a - b)
    return torch.abs(d)

def transform_local_to_global_angle(nz_vals: torch.Tensor, theta_local_vals: torch.Tensor) -> torch.Tensor:
    """
    使用 atan2 将局部角度 theta_local_vals 变换到全局坐标系。
    这是更稳健和正确的实现方式。
    """
    # 确保 nz_vals 广播到与 theta_local_vals 相同的形状
    if nz_vals.shape != theta_local_vals.shape:
        nz_vals = torch.broadcast_to(nz_vals, theta_local_vals.shape)

    return torch.atan2(torch.sin(theta_local_vals), nz_vals * torch.cos(theta_local_vals))

# ------------------ 对法向量场预计算不可达角区间（一次） ------------------
def compute_intervals_from_normals(nx: torch.Tensor,
                                   ny: torch.Tensor,
                                   nz: torch.Tensor,
                                   h: float = 8.0,
                                   min_edge: float = 10.0,
                                   max_edge: float = 20.0,
                                   device: Optional[torch.device] = None):
    """
    输入: nx,ny,nz (H,W) 栅格法向量（torch.tensor）
    输出: dict 包含扁平化的 interval arrays (s1..s5, e1..e5) 和掩码（向量化、长度 Ncells）
    这些 intervals 都是全局角度（rad）且已经被 normalize 到 [-pi, pi)。
    """
    if device is None:
        device = nx.device if isinstance(nx, torch.Tensor) else torch.device('cpu')

    if not isinstance(nx, torch.Tensor):
        nx = torch.tensor(nx, dtype=torch.float32, device=device)
        ny = torch.tensor(ny, dtype=torch.float32, device=device)
        nz = torch.tensor(nz, dtype=torch.float32, device=device)
    else:
        nx = nx.to(device).float(); ny = ny.to(device).float(); nz = nz.to(device).float()

    H, W = nx.shape
    N = H * W
    nx_f = nx.reshape(-1)
    ny_f = ny.reshape(-1)
    nz_f = nz.reshape(-1)

    # b 值及分类
    terrain_slope = torch.acos(torch.clip(nz_f, 0, 1))  # (N,) 计算坡度
    b_vals = h * torch.tan(terrain_slope)
    sqrt_term = torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device, dtype=torch.float32))

    mask_reachable = b_vals < min_edge
    mask_partial = (b_vals >= min_edge) & (b_vals < max_edge)
    mask_complex = (b_vals >= max_edge) & (b_vals < sqrt_term)
    mask_unreachable = b_vals >= sqrt_term

    # 预置 NaN interval 容器
    nan = torch.full((N,), float('nan'), device=device)
    s1 = nan.clone(); e1 = nan.clone()
    s2 = nan.clone(); e2 = nan.clone()
    s3 = nan.clone(); e3 = nan.clone()
    s4 = nan.clone(); e4 = nan.clone()
    s5 = nan.clone(); e5 = nan.clone()

    normal_proj = torch.atan2(ny_f, nx_f)  # (N,)

    # partial case
    if mask_partial.any():
        idx = mask_partial
        pb = b_vals[idx]
        p_nz = nz_f[idx]
        s1_vals = torch.asin(min_edge / pb)
        e1_vals = math.pi - s1_vals
        s2_vals = -s1_vals
        e2_vals = -math.pi + s1_vals

        normal_proj_idx = normal_proj[idx]
        s1_trans = transform_local_to_global_angle(p_nz, s1_vals)
        e1_trans = transform_local_to_global_angle(p_nz, e1_vals)
        s2_trans = transform_local_to_global_angle(p_nz, s2_vals)
        e2_trans = transform_local_to_global_angle(p_nz, e2_vals)

        s1[idx] = normalize_angle(normal_proj_idx + s1_trans)
        e1[idx] = normalize_angle(normal_proj_idx + e1_trans)
        s2[idx] = normalize_angle(normal_proj_idx + s2_trans)
        e2[idx] = normalize_angle(normal_proj_idx + e2_trans)

    # complex case
    if mask_complex.any():
        idx = mask_complex
        cb = b_vals[idx]; c_nz = nz_f[idx]
        r1_vals = torch.asin(min_edge / cb)
        r2_vals = torch.acos(max_edge / cb)

        s1v = -r2_vals; e1v = r2_vals
        s2v = r1_vals; e2v = torch.pi - r1_vals
        p1v = torch.pi - r2_vals
        p2v = -torch.pi + r2_vals
        s3v = -torch.pi + r1_vals; e3v = -r1_vals

        normal_proj_idx = normal_proj[idx]
        s1t = transform_local_to_global_angle(c_nz, s1v); e1t = transform_local_to_global_angle(c_nz, e1v)
        s2t = transform_local_to_global_angle(c_nz, s2v); e2t = transform_local_to_global_angle(c_nz, e2v)
        p1t = transform_local_to_global_angle(c_nz, p1v); p2t = transform_local_to_global_angle(c_nz, p2v)
        s3t = transform_local_to_global_angle(c_nz, s3v); e3t = transform_local_to_global_angle(c_nz, e3v)

        s1[idx] = normalize_angle(normal_proj_idx + s1t)
        e1[idx] = normalize_angle(normal_proj_idx + e1t)
        s2[idx] = normalize_angle(normal_proj_idx + s2t)
        e2[idx] = normalize_angle(normal_proj_idx + e2t)
        s3[idx] = normalize_angle(normal_proj_idx + s3t)
        e3[idx] = normalize_angle(normal_proj_idx + e3t)
        s4[idx] = normalize_angle(normal_proj_idx + p1t)
        e4[idx] = nan[idx]  # 表示 open-ended (> p1)
        s5[idx] = nan[idx]
        e5[idx] = normalize_angle(normal_proj_idx + p2t)

    intervals = [(s1, e1), (s2, e2), (s3, e3), (s4, e4), (s5, e5)]

    # H, W = nx.shape[0], nx.shape[1]
    # flat_idx_87_43 = 87 * W + 43
    # print(f"Debug for cell (87, 43):")
    # print(f"  - nx, ny, nz: {nx_f[flat_idx_87_43].item()}, {ny_f[flat_idx_87_43].item()}, {nz_f[flat_idx_87_43].item()}")
    # print(f"  - b_val: {b_vals[flat_idx_87_43].item()}")
    # print(f"  - Is Reachable: {mask_reachable[flat_idx_87_43].item()}")
    # print(f"  - Is Partial: {mask_partial[flat_idx_87_43].item()}")
    # print(f"  - Is Complex: {mask_complex[flat_idx_87_43].item()}")
    # print(f"  - Is Unreachable: {mask_unreachable[flat_idx_87_43].item()}")

    return {
        's_masks': (mask_reachable, mask_partial, mask_complex, mask_unreachable),
        'intervals': intervals,
        'b_vals': b_vals,
        'shape': (nx.shape[0], nx.shape[1])
    }

# ------------------ 单点查询（使用预计算 intervals） ------------------
def compute_esdf_at_point(nx, ny, nz,
                          x_q, y_q, yaw_q,
                          resolution: float = 0.1,
                          origin: tuple = (-5.0, -5.0),
                          yaw_weight: float = 0.5,
                          h: float = 8.0,
                          min_edge: float = 10.0,
                          max_edge: float = 20.0,
                          search_radius: float = 5.0,
                          device: Optional[torch.device] = None,
                          chunk_cells: int = 2000,
                          **batch_kwargs):
    """
    单点查询包装（基于 compute_esdf_batch 实现）。
    输入:
        nx,ny,nz: (H,W) 栅格法向量 torch.Tensor 或可转为 tensor
        x_q,y_q,yaw_q: 世界坐标点与朝向（yaw 单位：弧度）
        其余参数与 compute_esdf_batch 保持一致。
    返回:
        (esdf_val, (i,j) 或 None, yaw_dist 或 None)
    说明:
        - 内部将 (x_q,y_q,yaw_q) 封装为 shape (1,3) 的 queries，然后调用 compute_esdf_batch。
        - 若需要频繁单点查询并且想复用 intervals，请参见下方“性能提示”。
    """
    # 设备选择
    if device is None:
        device = nx.device if isinstance(nx, torch.Tensor) else torch.device('cpu')

    # 构造单条 query 的 tensor
    queries = torch.tensor([[float(x_q), float(y_q), float(yaw_q)]], dtype=torch.float32, device=device)

    # 调用 batch 版（它会返回长度为 1 的 results 列表）
    results = compute_esdf_batch(nx, ny, nz, queries,
                                 resolution=resolution,
                                 origin=origin,
                                 yaw_weight=yaw_weight,
                                 h=h, min_edge=min_edge, max_edge=max_edge,
                                 search_radius=search_radius,
                                 device=device,
                                 chunk_cells=chunk_cells,
                                 **batch_kwargs)

    # results[0] 是 (esdf_val, (i,j) or None, yaw_dist or None)
    return results[0]

# ------------------ 批量查询（复用 intervals，分块计算） ------------------
def compute_esdf_batch(nx: torch.Tensor,
                       ny: torch.Tensor,
                       nz: torch.Tensor,
                       queries: torch.Tensor,
                       resolution: float = 0.1,
                       origin: Tuple[float, float] = (-5.0, -5.0),
                       yaw_weight: float = 0.5,
                       h: float = 8.0,
                       min_edge: float = 10.0,
                       max_edge: float = 20.0,
                       search_radius: Optional[Union[float, Sequence[Optional[float]]]] = 5.0,
                       device: Optional[torch.device] = None,
                       chunk_cells: int = 2000 # 分块大小（每次处理的格点数）默认 2000
                    ):
    """
    批量查询（向量化 + 分块），返回列表：每项 (esdf_val, (i,j) or None, yaw_dist or None)
    关键点：角度比较使用 normalize_angle/周期性处理，intervals 预计算一次复用。
    """
    if device is None:
        device = nx.device if isinstance(nx, torch.Tensor) else torch.device('cpu')

    # convert to tensors on device
    if not isinstance(nx, torch.Tensor):
        nx = torch.tensor(nx, dtype=torch.float32, device=device)
        ny = torch.tensor(ny, dtype=torch.float32, device=device)
        nz = torch.tensor(nz, dtype=torch.float32, device=device)
    else:
        nx = nx.to(device).float(); ny = ny.to(device).float(); nz = nz.to(device).float()

    queries = queries.to(device).float()
    Nq = queries.shape[0]
    x_qs = queries[:, 0]; y_qs = queries[:, 1]; yaw_qs = queries[:, 2]

    H, W = nx.shape
    x0, y0 = origin
    res = resolution

    # 标准化 search_radius -> (Nq,)
    if search_radius is None:
        radius_qs = torch.full((Nq,), float('inf'), device=device)
    elif isinstance(search_radius, Sequence):
        if len(search_radius) != Nq:
            raise ValueError("search_radius length must equal number of queries")
        radius_qs = torch.tensor([float(r) if r is not None else float('inf') for r in search_radius],
                                  device=device, dtype=torch.float32)
    else:
        radius_qs = torch.full((Nq,), float(search_radius), device=device, dtype=torch.float32)

    # 哪些 query 在地图内
    inside_mask = (x_qs >= x0) & (x_qs <= x0 + W * res) & (y_qs >= y0) & (y_qs <= y0 + H * res)

    # grid centers flatten
    xs = x0 + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * res
    ys = y0 + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * res
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    Xc = grid_x.reshape(-1); Yc = grid_y.reshape(-1)
    Ncells = Xc.shape[0]

    # 预计算 intervals
    info = compute_intervals_from_normals(nx, ny, nz, h=h, min_edge=min_edge, max_edge=max_edge, device=device)
    intervals = info['intervals']
    mask_unreachable = info['s_masks'][3]  # (Ncells,)

    infv = float('inf')
    best_vals = torch.full((Nq,), infv, device=device)
    best_flat_idx = torch.full((Nq,), -1, dtype=torch.long, device=device)
    best_yaw_dist = torch.full((Nq,), infv, device=device)

    # 分块遍历格点，避免内存峰值
    num_chunks = (Ncells + chunk_cells - 1) // chunk_cells
    for c in range(num_chunks):
        s_idx = c * chunk_cells
        e_idx = min((c + 1) * chunk_cells, Ncells)
        idxs = torch.arange(s_idx, e_idx, device=device, dtype=torch.long)
        Xc_chunk = Xc[idxs]  # (M,)
        Yc_chunk = Yc[idxs]

        # compute d_xy_chunk (Nq, M)
        dx = Xc_chunk.reshape(1, -1) - x_qs.reshape(-1, 1)
        dy = Yc_chunk.reshape(1, -1) - y_qs.reshape(-1, 1)
        d_xy_chunk = torch.sqrt(dx * dx + dy * dy)

        # mask_search per query
        radius_chunk = radius_qs.reshape(-1, 1)
        mask_search_chunk = d_xy_chunk <= radius_chunk  # (Nq,M)

        M = idxs.shape[0]
        yaw_dist_chunk = torch.full((Nq, M), infv, device=device)

        # 对每个 interval 计算 (Nq, M) 的 yaw 距离（周期化处理）
        for s_arr, e_arr in intervals:
            s_chunk = s_arr[idxs]  # (M,)
            e_chunk = e_arr[idxs]
            # normalize to [-pi,pi)
            s_norm = normalize_angle(s_chunk).reshape(1, M)
            e_norm = normalize_angle(e_chunk).reshape(1, M)
            yaw = normalize_angle(yaw_qs).reshape(Nq, 1)  # (Nq,1)

            # valid intervals mask
            valid_mask = (~torch.isnan(s_chunk)) & (~torch.isnan(e_chunk))  # (M,)
            if valid_mask.any():
                # 计算到端点的角距离（Nq, M）
                ds = torch.abs(normalize_angle(yaw - s_norm))  # (Nq,M)
                de = torch.abs(normalize_angle(yaw - e_norm))
                dmin = torch.minimum(ds, de)

                # normal / crossing intervals（按列处理，并同时限制到 valid 列）
                normal = (s_norm <= e_norm).squeeze(0)  # (M,)
                normal_valid = normal & valid_mask
                if normal_valid.any():
                    inside_n = (yaw >= s_norm) & (yaw <= e_norm)  # (Nq,M)
                    sel = normal_valid
                    yaw_dist_chunk[:, sel] = torch.where(inside_n[:, sel],
                                                         torch.zeros_like(dmin[:, sel]),
                                                         dmin[:, sel])

                comp_valid = (~normal) & valid_mask
                if comp_valid.any():
                    inside_c = (yaw >= s_norm) | (yaw <= e_norm)
                    sel = comp_valid
                    yaw_dist_chunk[:, sel] = torch.where(inside_c[:, sel],
                                                         torch.zeros_like(dmin[:, sel]),
                                                         dmin[:, sel])

            # 非有效 interval 的位置保持 inf

        # fully unreachable -> yaw_dist = 0
        mask_unreach_chunk = mask_unreachable[idxs]  # (M,)
        if mask_unreach_chunk.any():
            yaw_dist_chunk[:, mask_unreach_chunk] = 0.0

        # 合成距离并屏蔽不在搜索范围处
        combined_chunk = torch.sqrt(d_xy_chunk**2 + (yaw_weight * yaw_dist_chunk)**2)
        combined_chunk = torch.where(mask_search_chunk, combined_chunk, torch.full_like(combined_chunk, infv))

        # per-query chunk 最小值
        chunk_min_vals, chunk_argmins = torch.min(combined_chunk, dim=1)  # (Nq,), (Nq,)
        better_mask = chunk_min_vals < best_vals
        if better_mask.any():
            best_vals[better_mask] = chunk_min_vals[better_mask]
            best_flat_idx[better_mask] = (s_idx + chunk_argmins[better_mask])
            # 更新 yaw dist：需要 gather per query
            q_idx = torch.nonzero(better_mask, as_tuple=False).squeeze(-1)
            best_yaw_dist[better_mask] = yaw_dist_chunk[q_idx, chunk_argmins[better_mask]]

    # 组织结果，格式为 (esdf_val, (i,j) or None, yaw_dist or None)
    # esdf_val 是合成距离，若无效则为 inf
    # (i,j) 是最近不可达格点索引，若无效则为 None；
    # yaw_dist 是角度距离，若为0表示不可达，若大于0则为距离最近的不可达区间边界的角度距离。
    results = []
    # 用 tensor 表示无效的 inf 值，保持设备一致
    inf_tensor = torch.tensor(float('inf'), device=device, dtype=torch.float32)
    for q in range(Nq):
        # 使用 .item() 仅做布尔/索引判断（不会影响梯度图）
        if not bool(inside_mask[q].item()):
            # 返回张量形式的 esdf（inf）和 None 表示无索引/yaw
            results.append((inf_tensor, None, None))
            continue
        if int(best_flat_idx[q].item()) < 0:
            results.append((inf_tensor, None, None))
        else:
            # 保留 esdf 和 yaw_dist 为 tensor（保留梯度链条）
            esdf_val = best_vals[q]         # torch scalar tensor (可导)
            yaw_dist = best_yaw_dist[q]     # torch scalar tensor (可导)
            # 索引仍以 Python int 返回（不参与微分），如需 tensor 可改为 torch.tensor(...)
            flat = int(best_flat_idx[q].item())
            i = int(flat // W)
            j = int(flat % W)
            results.append((esdf_val, (i, j), yaw_dist))
    return results

def query_is_unreachable_by_match(result_tuple, x_q, y_q, yaw_q, origin=(-5.0, -5.0), resolution=0.1, allow_neighbor_offset: int = 0):
    """
    基于 compute_esdf_* 的返回 (esdf_val, (i,j) or None, yaw_dist or None) 判断查询位姿是否不可达。
    逻辑：
      - 先把 (x_q,y_q) 映射到栅格 (i_expected,j_expected)
      - 若返回的 (i,j) 与期望一致（允许小偏差 allow_neighbor_offset），且 yaw_dist == 0，则认为不可达
    参数:
      - result_tuple: compute_esdf 返回的单条结果
      - allow_neighbor_offset: 允许的行/列离差（0 表示严格相等，1 表示允许相邻格子）
    返回:
      - bool (True = 不可达), dict (调试信息)
    备注:
      - 该判定是基于“最近不可达格点恰好就是查询所在格点且角距为0”的启发式判断。
      - 若需严格判断“查询角度是否落入该格点的不可达角区间”，请配合 compute_intervals_from_normals + is_yaw_in_intervals。
    """
    esdf_val, ij, yaw_dist = result_tuple
    x0, y0 = origin

    # 先处理简单情况
    if ij is None:
        return False, {'reason': 'no_nearest_unreachable_returned', 'esdf': esdf_val}

    # 计算期望栅格索引（用 floor 以避免截断负数问题）
    import math
    j_expected = int(math.floor((float(x_q) - float(x0)) / float(resolution)))
    i_expected = int(math.floor((float(y_q) - float(y0)) / float(resolution)))

    i_ret, j_ret = ij

    # 比较是否一致（允许邻近偏差）
    matches = (abs(int(i_ret) - i_expected) <= allow_neighbor_offset) and (abs(int(j_ret) - j_expected) <= allow_neighbor_offset)

    is_unreach = bool(matches and (yaw_dist is not None) and (float(yaw_dist) == 0.0))

    info = {
        'i_expected': i_expected, 'j_expected': j_expected,
        'i_ret': int(i_ret), 'j_ret': int(j_ret),
        'matches': matches,
        'yaw_dist': float(yaw_dist) if yaw_dist is not None else None,
        'esdf': esdf_val
    }
    return is_unreach, info

def query_is_unreachable_by_match_batch(results, queries, origin=(-5.0, -5.0), resolution=0.1, allow_neighbor_offset: int = 0):
    """
    批量版：基于 compute_esdf_batch 返回的 results 列表和对应 queries（N x 3）判断每个位姿是否不可达。
    返回 (flags, infos)：
      - flags: list of bool，True 表示不可达（匹配且 yaw_dist == 0）
      - infos: list of dict，包含 i_expected/j_expected, i_ret/j_ret, matches, yaw_dist, esdf
    参数说明同单条版本；queries 支持 torch.Tensor 或可迭代的 (x,y,yaw) 序列。
    """
    import math
    # 规范 queries 到 numpy list of tuples
    if hasattr(queries, "device") and isinstance(queries, torch.Tensor):
        q_np = queries.detach().cpu().numpy()
    else:
        q_np = list(queries)

    Nq = len(q_np)
    if len(results) != Nq:
        raise ValueError(f"results length ({len(results)}) != queries length ({Nq})")

    x0, y0 = origin
    flags = []
    infos = []
    for idx in range(Nq):
        res = results[idx]
        # res 应为 (esdf_val, (i,j) or None, yaw_dist or None)
        esdf_val, ij, yaw_dist = res
        # extract x,y from q_np row/tuple
        x_q = float(q_np[idx][0])
        y_q = float(q_np[idx][1])

        if ij is None:
            info = {'reason': 'no_nearest_unreachable_returned', 'esdf': esdf_val}
            flags.append(False); infos.append(info); continue

        j_expected = int(math.floor((x_q - float(x0)) / float(resolution)))
        i_expected = int(math.floor((y_q - float(y0)) / float(resolution)))
        i_ret, j_ret = ij

        matches = (abs(int(i_ret) - i_expected) <= allow_neighbor_offset) and (abs(int(j_ret) - j_expected) <= allow_neighbor_offset)
        is_unreach = bool(matches and (yaw_dist is not None) and (float(yaw_dist) == 0.0))

        info = {
            'i_expected': i_expected, 'j_expected': j_expected,
            'i_ret': int(i_ret), 'j_ret': int(j_ret),
            'matches': matches,
            'yaw_dist': float(yaw_dist) if yaw_dist is not None else None,
            'esdf': esdf_val
        }
        flags.append(is_unreach)
        infos.append(info)

    return flags, infos

# 在你的脚本里引入或粘贴 —— 基于之前的 compute_intervals_from_normals 函数
def debug_intervals_stats(nx, ny, nz, h=8.0, min_edge=10.0, max_edge=20.0, device=None):
    if device is None:
        device = nx.device if isinstance(nx, torch.Tensor) else torch.device('cpu')
    if not isinstance(nx, torch.Tensor):
        nx = torch.tensor(nx, dtype=torch.float32, device=device)
        ny = torch.tensor(ny, dtype=torch.float32, device=device)
        nz = torch.tensor(nz, dtype=torch.float32, device=device)
    else:
        nx = nx.to(device).float(); ny = ny.to(device).float(); nz = nz.to(device).float()

    H, W = nx.shape
    nx_f = nx.reshape(-1); ny_f = ny.reshape(-1); nz_f = nz.reshape(-1)
    terrain_slope = torch.atan2(torch.sqrt(nx_f**2 + ny_f**2), torch.abs(nz_f))
    b_vals = h * torch.tan(terrain_slope)
    sqrt_term = torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device, dtype=torch.float32))

    mask_reachable = (b_vals < min_edge).sum().item()
    mask_partial = ((b_vals >= min_edge) & (b_vals < max_edge)).sum().item()
    mask_complex = ((b_vals >= max_edge) & (b_vals < sqrt_term)).sum().item()
    mask_unreachable = (b_vals >= sqrt_term).sum().item()

    print(f"地图尺寸: {H}x{W} => 总格子 {H*W}")
    print(f"b_vals: min={float(b_vals.min().item()):.4g}, max={float(b_vals.max().item()):.4g}, mean={float(b_vals.mean().item()):.4g}")
    print("各类别格子数量:")
    print(f"  reachable (<{min_edge}): {mask_reachable}")
    print(f"  partial  (>= {min_edge} and < {max_edge}): {mask_partial}")
    print(f"  complex  (>= {max_edge} and < sqrt(min^2+max^2)): {mask_complex}")
    print(f"  unreachable (>= sqrt(min^2+max^2)): {mask_unreachable}")

# ------------------ 快速示例（测试周期性） ------------------
if __name__ == "__main__":
    # # 创建示例场 (100x100)
    # H, W = 100, 100
    # device = torch.device('cpu')
    # nx = torch.randn(H, W, device=device) * 0.05
    # ny = torch.randn(H, W, device=device) * 0.05
    # nz = torch.sqrt(torch.clamp(1.0 - nx**2 - ny**2, min=0.0))
    
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    
    # 从数据集中加载法向量场
    from dataLoader_uneven import UnevenPathDataLoader
    # 测试加载数据集
    env_list = ['env000014']
    dataFolder = '/home/yrf/MPT/data/terrain_dataset/val'
    dataset = UnevenPathDataLoader(env_list, dataFolder)
    # 测试数据集idx=0的返回值
    path_index = 1
    sample = dataset[path_index]
    print(sample['map'].shape)
    print(sample['anchor'].shape)
    print(sample['labels'].shape)
    
    nx = sample['map'][0, :, :].to(torch.float32)  # 假设 map 是 (1,1,H,W) 形状
    ny = sample['map'][1, :, :].to(torch.float32)  # 假设 map 是 (1,1,H,W) 形状
    nz = sample['map'][2, :, :].to(torch.float32)  # 假设 map 是 (1,1,H,W) 形状
    
    H, W = nx.shape
    d_safe = 0.  # 安全距离（米）
    kalpa = 0.6 # 安全损失衰减速率
    print(f"安全距离: {d_safe} 米, 衰减速率: {kalpa}")

    trajectory = sample['trajectory'].to(torch.float32)  # trajectory 是 (K,3) 形状
    x_opt = trajectory[:, 0]  # x 坐标
    y_opt = trajectory[:, 1]  # y 坐标
    yaw_opt = trajectory[:, 2]  # yaw 角度（弧度）

    # 两个 query 测试 -pi/pi 周期性
    # queries = torch.tensor([
    #     [3.0, 0.0, math.pi - 0.01],   # 接近 +pi
    #     [-3.0, 0.0, -math.pi + 0.01],  # 接近 -pi
    # ], dtype=torch.float32)
    
    queries = torch.stack([
        x_opt[:2],  # 前两个 x 坐标
        y_opt[:2],  # 前两个 y 坐标
        yaw_opt[:2]  # 前两个 yaw 角度
    ], dim=1).to(device)  # shape (2, 3)
    
    import numpy as np
    import time
    start_time = time.time()
    res = compute_esdf_batch(nx, ny, nz, queries,
                             resolution=0.1, origin=(-5.0, -5.0),
                             yaw_weight=0.4, search_radius=5.0, chunk_cells=3000, device=device)
    end_time = time.time()
    print(f"批量查询耗时: {end_time - start_time:.4f} 秒")
    is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-5.0, -5.0), resolution=0.1)
    for q, r, is_unreachable, info in zip(queries, res, is_unreachables, infos):
        # print("nx at query:", nx[int(q[1].item() / 0.1 + 50), int(q[0].item() / 0.1 + 50)].item())
        # print("ny at query:", ny[int(q[1].item() / 0.1 + 50), int(q[0].item() / 0.1 + 50)].item())
        # print("nz at query:", nz[int(q[1].item() / 0.1 + 50), int(q[0].item() / 0.1 + 50)].item())
        print("query:", q.tolist(), "->", r)
        print("  ==  safety loss:", torch.exp(-(r[0] - d_safe) / kalpa))
        print("  ==  is unreachable:", is_unreachable)
        print("  ==  info:", info)
        
    # 单点查询测试  
    start_time = time.time()
    res1 = compute_esdf_at_point(nx, ny, nz,
                                 x_q=queries[0, 0].item(),
                                 y_q=queries[0, 1].item(),
                                 yaw_q=queries[0, 2].item(),
                                 resolution=0.1, origin=(-5.0, -5.0),
                                 yaw_weight=0.4, search_radius=5.0, device=device)
    end_time = time.time()
    print(f"单点查询耗时: {end_time - start_time:.4f} 秒")
    print("单点查询结果1:", res1)
    # 再次单点查询
    start_time = time.time()
    res2 = compute_esdf_at_point(nx, ny, nz,
                                 x_q=queries[1, 0].item(),
                                 y_q=queries[1, 1].item(),
                                 yaw_q=queries[1, 2].item(),
                                 resolution=0.1, origin=(-5.0, -5.0),
                                 yaw_weight=0.4, search_radius=5.0, device=device)
    end_time = time.time()
    print(f"单点查询耗时: {end_time - start_time:.4f} 秒")
    print("单点查询结果2:", res2)

    debug_intervals_stats(nx, ny, nz, h=8.0, min_edge=10.0, max_edge=20.0)
# ------------------ 结束 ------------------
