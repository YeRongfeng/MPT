# Abstract Outline

> 会议后重构版本。摘要先让普通机器人审稿人理解任务和困难，再引入方法术语；mask 是支撑设计，轨迹表征与二阶段训练是两项核心创新。

## 叙事顺序

1. **实际任务**
   - 崎岖地形上的无人地面车辆需要一条能够到达目标，同时避开不可通行地形、倾覆风险和过急转弯的全局路径。
   - 开头不出现渐进式建图、固定输入窗口、可行域或梯度冲突等术语。

2. **两个直观困难**
   - 直接预测整条路径时，精确的起终点位置和朝向仍要依赖网络学习。
   - 模仿专家路线能够学习如何绕行，却不能直接协调具体地形上的多项物理要求。

3. **方法总述**
   - 提出 Path MeanFlow：根据当前可用地形，一次前向生成一条全局路径。
   - 此处只给方法定位，不立即解释网络结构。

4. **核心创新一：轨迹表征**
   - 将已知起终点位置和朝向解析写入样条解码器。
   - 网络只生成尚未确定的内部路线形状。

5. **核心创新二：二阶段训练**
   - 第一阶段从专家路径学习长距离路线结构。
   - 第二阶段以仅在训练期可用的完整地形为监督，协调规划区域、倾覆风险和曲率目标。
   - 不声称三个目标必然共同下降或同时收敛。

6. **支撑设计与部署边界**
   - mask 告诉生成器当前观测中哪些区域可以用于规划，但不作为第三项核心创新。
   - 部署只读取带掩码的当前观测，一次前向生成一条路径，不访问完整地形，也不执行在线物理代价优化。

7. **实验结果**
   - 首先报告未见地形仿真中相对实际评测里表现最好的 partial-information baseline 的 Feasible@1。
   - 再报告 PMTA 相对专家路线学习阶段的变化、路线分布代价和 p95 延迟。
   - 结果冻结前保留占位符，不使用 safer、superior 或 real-time 等结论词。

## 英文句子合同

1. Plain-language task and desired path behavior.
2. Two limitations of direct path prediction and imitation learning.
3. Path MeanFlow as a one-pass global path generator.
4. Boundary-aligned trajectory representation.
5. Expert-route learning followed by PMTA.
6. Mask semantics and deployment information boundary.
7. Simulation evidence with frozen numerical results.
