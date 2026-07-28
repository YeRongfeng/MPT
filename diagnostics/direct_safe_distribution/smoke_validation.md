# 实现 smoke validation

- canonicalization/全局硬边界包装：8/8 `unittest` 通过；
- split/manifest：4/4 `unittest` 通过，实际 100 图 hash 复核通过；训练/
  teacher 的 split-aware 校验不会读取 test map；
- tiny DiT canonical MeanFlow：CPU forward/JVP/backward 成功，梯度均有限；
- real 146M 参数 DiT：RTX 3060 6 GB 上完成 1 epoch、70 train + 15
  validation 样本，未 OOM；
- 共同 ±28.285 m 画布的原地图有效采样比例约 12.25%；A/B/C/D 相同承担该
  分辨率代价，报告必须把它作为控制变量和潜在性能上限；
- safety teacher 数据链：1 map × 1 condition × K=8、真实 yaw-aware ESDF、
  topology trust projection 和完整 NPZ/CSV metadata 成功；
- evaluator：K=32、Safe@K/Valid Rate/Oracle/mode/dedup/latency/memory 全链路
  成功；
- SE(2) evaluator：当前真实 Stage‑1 checkpoint 的多角度曲线成功。

这些只验证接口和资源可行性，不替代 3 seeds × A/B/C/D 的完整预算实验。
