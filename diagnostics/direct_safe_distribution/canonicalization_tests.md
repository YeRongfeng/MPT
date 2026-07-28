# Canonicalization 测试记录

运行环境：`conda env vim`，标准库 `unittest`。

通过的断言：

1. pose/trajectory canonicalize→decanonicalize 往返；
2. canonical 起点为 `(0,0)`、终点位于正 x 轴；
3. yaw 始终 wrap 到 `[-pi,pi)`；
4. residual 旋转后保持零和并可逆；
5. 26 控制点 residual decode 与三次 B 样条稠密曲线均与旋转交换；
6. raster 空间逆映射、`(nx,ny)` 旋转、`nz`/elevation 标量处理正确；
7. 旋转后的同一 source 在 task frame 中恢复相同数值；
8. `enabled=False` 返回原 tensor（相同 data pointer、逐位相同）；
9. 小型 `PathDiffusionTransformer` 在 disabled preprocessing 前后输出逐位相同；
10. 共同输出包装把极端 canonical residual 恢复到原地图硬边界并保持端点；
11. split 稳定、三组无交集、manifest tamper 可检测；
12. train/validation 校验不会打开 test map，完整校验仍能发现 test 被篡改。

结果：canonicalization/输出边界 8 项测试和 split 4 项测试全部通过。连续点/向量断言使用
解析容差；raster 测试单独验证重采样，未将离散栅格宣称为数值严格等变。
