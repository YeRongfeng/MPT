"""
eval_model_car.py - 车辆模型路径规划评估脚本

【核心功能】
本脚本专门用于评估基于车辆动力学模型的路径规划系统性能。与标准的点机器人路径规划不同，
本脚本考虑了车辆的运动学约束，使用SST（Stable Sparse Tree）规划器进行非完整约束下的路径规划。

【技术特点】
1. 车辆动力学建模：考虑车辆长度、转向角度等物理约束
2. 非完整约束规划：处理车辆无法任意方向移动的约束
3. MPT引导规划：使用Transformer预测的关键区域指导路径搜索
4. 碰撞检测优化：针对车辆几何形状的碰撞检测
5. 性能评估框架：全面的规划成功率、时间、质量评估

【应用场景】
1. 自动驾驶路径规划：城市环境中的车辆导航
2. 机器人车辆控制：仓储、物流等场景的移动机器人
3. 算法性能对比：MPT vs 传统规划算法的效果评估
4. 实际部署验证：真实环境下的规划算法验证

【系统架构】
输入：地图、起点、终点 → MPT预测 → 引导区域生成 → SST规划器 → 路径输出 → 性能评估

【技术创新】
1. 混合规划策略：结合学习引导和传统规划的优势
2. 动态掩码机制：根据MPT预测动态调整搜索空间
3. 探索-利用平衡：在引导搜索和自由探索间切换
4. 多指标评估：时间、成功率、路径质量的综合评估

技术栈：
- OMPL (Open Motion Planning Library) 运动规划库
- PyTorch 深度学习框架
- scikit-image 图像处理
- NumPy 数值计算

参考文献：
- SST: A Path-Integral Approach to Trajectory Optimization
- Motion Planning Transformers: Long-Range Path Planning
"""

# 【核心依赖库导入】
import torch  # PyTorch深度学习框架：用于模型推理和张量计算
import torch.nn as nn  # 神经网络模块：提供网络层定义
import torch.nn.functional as F  # 函数式接口：提供激活函数、损失函数等
import skimage.io  # 图像IO操作：读取和保存地图图像
import skimage.morphology as skim  # 形态学操作：用于碰撞检测的图像膨胀处理
import numpy as np  # 数值计算库：矩阵运算和数组处理
import pickle  # 序列化库：保存和加载实验数据

from os import path as osp  # 路径操作：文件和目录路径处理
import argparse  # 命令行参数解析：处理脚本运行参数
import json  # JSON处理：读取模型配置文件

from functools import partial  # 函数工具：创建偏函数，用于参数绑定

# 【OMPL运动规划库导入】
# OMPL是开源运动规划库，提供多种规划算法和状态空间定义
try:
    from ompl import base as ob  # 基础模块：状态空间、有效性检查等核心组件
    # from ompl import geometric as og  # 几何规划：RRT*等几何路径规划算法（本脚本未使用）
    from ompl import control as oc  # 控制规划：处理动力学约束的规划算法（SST等）
    from ompl import util as ou  # 工具模块：日志、随机数生成等辅助功能
except ImportError:
    # 【依赖检查】确保运行环境安装了OMPL库
    raise ImportError("Container does not have OMPL installed")

# 【项目模块导入】
from transformer import Models  # Transformer模型：MPT路径规划网络
from utils import geom2pix  # 坐标转换工具：几何坐标到像素坐标的转换
from math import sin, cos, tan, pi  # 数学函数：车辆运动学计算所需的三角函数
from pathlib import Path  # 路径处理：现代化的路径操作接口

from dataLoader import get_encoder_input  # 数据加载器：准备模型输入数据
# 【全局参数配置】
res = 0.05  # 地图分辨率：每个像素代表的实际距离（米），影响坐标转换精度
dist_resl = res  # 距离分辨率：与地图分辨率保持一致，用于距离计算
length = 24  # 地图实际长度：24米×24米的正方形地图，定义了规划空间的边界
robot_radius = 0.2  # 车辆半径：0.2米，用于碰撞检测时的安全边距计算
carLength = 0.3  # 车辆轴距：0.3米，影响车辆转向特性和运动学模型


def pix2geom(pos, res=0.05, length=24):
    """
    像素坐标到几何坐标的转换函数
    
    【核心功能】
    将图像像素坐标系转换为实际的几何坐标系。像素坐标系原点在左上角，
    而几何坐标系原点在左下角，需要进行Y轴翻转。
    
    【坐标系转换原理】
    1. X轴转换：直接按分辨率缩放
    2. Y轴转换：需要翻转，因为像素Y轴向下，几何Y轴向上
    3. 转换公式：
       - geo_x = pix_x * res
       - geo_y = length - pix_y * res
    
    Args:
        pos (tuple): 像素坐标 (x, y)
            格式：(列索引, 行索引)
            范围：x ∈ [0, width-1], y ∈ [0, height-1]
            含义：在图像中的像素位置
        res (float): 地图分辨率，默认0.05米/像素
            作用：控制像素到实际距离的转换比例
            影响：更小的分辨率提供更高的定位精度
        length (float): 地图的实际长度，默认24米
            用途：Y轴翻转时的参考长度
            约束：应该等于地图高度 * res
    
    Returns:
        tuple: 几何坐标 (x, y)
            单位：米
            坐标系：右手坐标系，原点在左下角
            范围：x ∈ [0, length], y ∈ [0, length]
    
    【使用场景】
    1. 路径点转换：将规划器输出的像素路径转为实际坐标
    2. 起点终点设置：将用户指定的像素位置转为规划坐标
    3. 碰撞检测：将几何查询点转为图像索引
    4. 可视化显示：在实际坐标系中显示规划结果
    """
    return (pos[0]*res, length-pos[1]*res)  # X轴直接缩放，Y轴翻转后缩放


# 【网络相关参数】
receptive_field = 32  # 感受野大小：每个预测锚点影响的像素范围，决定patch区域大小
hashTable = [(20*r+4, 20*c+4) for c in range(24) for r in range(24)]  # 哈希表：将1D token索引映射到2D图像位置的查找表

# 【哈希表构建逻辑】
# - 24x24的网格：对应Transformer输出的token数量（576个token）
# - (20*r+4, 20*c+4)：每个token对应20x20像素区域，偏移4像素对齐
# - 这个映射关系与transformer/Models.py中的patch_embedding网络结构对应
# - 偏移量4：确保patch中心与网格中心对齐，避免边界效应



# 【OMPL规划参数配置】
# 
# 【状态空间定义】SE(2)状态空间
# SE(2)表示2D平面上的刚体运动，包含位置(x,y)和朝向(θ)
space = ob.SE2StateSpace()  # 创建SE(2)状态空间：支持车辆的位置和朝向

# 【位置边界设置】
# 限制车辆在地图范围内运动，防止越界
bounds = ob.RealVectorBounds(2)  # 创建2D边界对象：限制x,y坐标范围
bounds.setLow(0)  # 设置下界：x,y坐标的最小值为0
bounds.setHigh(length)  # 设置上界：x,y坐标的最大值为地图长度(24米)
space.setBounds(bounds)  # 将边界应用到状态空间

# 【控制空间定义】
# 定义车辆的控制输入：速度和转向角
cspace = oc.RealVectorControlSpace(space, 2)  # 创建2维控制空间：[速度, 转向角]

# 【控制输入边界】
# 限制车辆的控制输入范围，确保物理可行性
cbounds = ob.RealVectorBounds(2)  # 创建控制边界对象
cbounds.setLow(0, 0.0)   # 速度下界：0 m/s（不能倒车）
cbounds.setHigh(0, 1)    # 速度上界：1 m/s（最大前进速度）
cbounds.setLow(1, -.5)   # 转向角下界：-0.5 rad（最大左转）
cbounds.setHigh(1, .5)   # 转向角上界：0.5 rad（最大右转）
cspace.setBounds(cbounds)  # 将控制边界应用到控制空间

# 【规划器设置对象】
ss = oc.SimpleSetup(cspace)  # 创建简化设置对象：集成状态空间、控制空间和规划器
si = ob.SpaceInformation(space)  # 创建空间信息对象：提供状态空间的查询接口

# 【技术说明】
# 1. SE(2)空间：Special Euclidean Group，表示2D刚体变换
#    - 状态：(x, y, θ) 其中θ为朝向角
#    - 适用于：车辆、移动机器人等有朝向的对象
# 
# 2. 控制空间：定义如何改变状态
#    - 控制输入：[线速度, 角速度]
#    - 物理约束：速度和转向角的合理范围
# 
# 3. 边界设置：
#    - 状态边界：限制车辆活动区域
#    - 控制边界：限制车辆运动能力
#    - 安全考虑：防止不合理的控制输入

def kinematicCarODE(q, u, qdot):
    """
    车辆运动学微分方程
    
    【核心功能】
    定义车辆的运动学模型，描述控制输入如何影响车辆状态的变化率。
    基于自行车模型（bicycle model），这是车辆运动学建模的经典方法。
    
    【数学模型】
    车辆运动学方程：
    ẋ = v * cos(θ)     # X方向速度分量
    ẏ = v * sin(θ)     # Y方向速度分量  
    θ̇ = v * tan(δ) / L # 角速度，δ为前轮转向角，L为轴距
    
    【物理意义】
    1. 位置变化：车辆沿当前朝向移动
    2. 朝向变化：由前轮转向角和车辆速度决定
    3. 非完整约束：车辆不能侧向移动（侧滑忽略）
    
    Args:
        q (array): 当前状态 [x, y, θ]
            x, y: 车辆后轴中心的位置坐标（米）
            θ: 车辆朝向角，相对于X轴的角度（弧度）
        u (array): 控制输入 [v, δ]
            v: 线速度，车辆前进速度（米/秒）
            δ: 前轮转向角，相对于车辆纵轴的角度（弧度）
        qdot (array): 状态变化率输出 [ẋ, ẏ, θ̇]
            函数通过修改此数组返回状态导数
    
    【自行车模型假设】
    1. 车辆为刚体，不考虑悬挂系统
    2. 前轮负责转向，后轮负责驱动
    3. 忽略轮胎侧滑，车辆沿纵轴方向运动
    4. 低速运动，忽略动力学效应
    
    【参数说明】
    - carLength: 车辆轴距，影响转向灵敏度
    - 较长轴距：转向半径大，稳定性好
    - 较短轴距：转向灵活，但稳定性差
    """
    theta = q[2]  # 提取当前朝向角：车辆相对于X轴的角度
    
    # 【运动学方程实现】
    qdot[0] = u[0] * cos(theta)  # X方向位置变化率：速度在X轴的投影
    qdot[1] = u[0] * sin(theta)  # Y方向位置变化率：速度在Y轴的投影  
    qdot[2] = u[0] * tan(u[1]) / carLength  # 朝向变化率：基于自行车模型的转向公式
    
    # 【公式解释】
    # qdot[2] = v * tan(δ) / L
    # - v: 线速度 u[0]
    # - δ: 转向角 u[1] 
    # - L: 轴距 carLength
    # - tan(δ): 转向角的正切值，体现转向的非线性特性

class ValidityChecker(ob.StateValidityChecker):
    """
    ValidityChecker - 车辆状态有效性检查器
    
    【系统概述】
    继承自OMPL的StateValidityChecker，专门用于检查车辆状态是否与障碍物碰撞。
    考虑了车辆的几何尺寸，通过图像膨胀操作实现高效的碰撞检测。
    
    【核心功能】
    1. 碰撞检测：检查车辆在给定位置是否与障碍物碰撞
    2. 几何膨胀：考虑车辆尺寸，对障碍物进行膨胀处理
    3. 掩码集成：结合MPT预测的引导区域进行约束
    4. 边界检查：确保车辆不超出地图边界
    
    【技术特点】
    1. 高效检测：基于图像的快速碰撞检测
    2. 几何精确：考虑车辆实际尺寸的膨胀处理
    3. 智能引导：集成MPT预测的可行区域
    4. 边界安全：完整的边界条件检查
    
    【碰撞检测原理】
    1. 障碍物膨胀：将障碍物按车辆半径膨胀
    2. 点查询：将车辆简化为点进行碰撞检测
    3. 掩码过滤：仅在MPT允许的区域内搜索
    4. 快速查表：O(1)时间复杂度的碰撞查询
    
    在MPT系统中的定位：
    - 安全约束器：确保规划路径的安全性
    - 搜索引导器：结合学习预测指导搜索
    - 效率优化器：通过预处理提升检测效率
    """
    
    def __init__(self, si, CurMap, MapMask=None, res=0.05, robot_radius=robot_radius):
        """
        初始化车辆状态有效性检查器
        
        【核心功能】
        构建考虑车辆几何尺寸和MPT引导的碰撞检测系统。
        通过图像形态学操作预处理地图，实现高效的碰撞查询。
        
        【初始化流程】
        1. 地图预处理：反转地图，将自由空间标记为1
        2. 几何膨胀：按车辆半径膨胀障碍物
        3. 掩码集成：结合MPT预测的可行区域
        4. 查找表构建：生成最终的有效性查找表
        
        Args:
            si (ob.SpaceInformation): OMPL空间信息对象
                作用：提供状态空间的基本信息和接口
                用途：父类初始化和状态查询
            CurMap (np.array): 当前地图数组
                格式：2D numpy数组，0表示障碍物，1表示自由空间
                尺寸：通常为正方形，如480x480像素
                数据类型：float或int，值域[0,1]
            MapMask (np.array, optional): MPT预测的掩码区域
                格式：与CurMap相同尺寸的布尔数组
                含义：True表示MPT认为可行的区域
                用途：引导搜索，提高规划效率
                默认：None表示不使用MPT引导
            res (float): 地图分辨率，默认0.05米/像素
                作用：控制膨胀操作的实际尺寸
                影响：影响碰撞检测的精度和安全边距
            robot_radius (float): 车辆半径，用于碰撞检测
                含义：车辆外接圆的半径
                用途：确定障碍物膨胀的尺寸
                安全考虑：通常设置得略大于实际尺寸
        
        【图像处理流程】
        1. 地图反转：InvertMap = 1 - CurMap
           - 目的：将障碍物标记为1，便于膨胀操作
           - 结果：障碍物区域为1，自由空间为0
        
        2. 障碍物膨胀：使用形态学膨胀操作
           - 膨胀核：圆形，半径为0.1米（约为车辆半径）
           - 目的：考虑车辆尺寸，扩大障碍物范围
           - 效果：原本可通过的狭窄区域变为不可通过
        
        3. 地图恢复：MapDilate = 1 - InvertMapDilate  
           - 目的：恢复原始的地图表示（1为自由空间）
           - 结果：膨胀后的自由空间地图
        
        4. 掩码集成：
           - 无掩码：直接使用膨胀后的地图
           - 有掩码：取膨胀地图与MPT掩码的交集
           - 效果：仅在MPT认为可行的区域内搜索
        """
        super().__init__(si)  # 调用父类构造函数：初始化OMPL状态有效性检查器
        self.size = CurMap.shape  # 保存地图尺寸：用于边界检查和坐标转换
        
        # 【步骤1】地图反转：将障碍物标记为1，自由空间标记为0
        InvertMap = np.abs(1-CurMap)  # 反转地图：0->1（障碍物），1->0（自由空间）
        
        # 【步骤2】障碍物膨胀：考虑车辆半径，扩大障碍物区域
        # 使用圆形结构元素进行形态学膨胀操作
        InvertMapDilate = skim.dilation(InvertMap, skim.disk((0.1)/res))  # 膨胀障碍物：半径为0.1米
        
        # 【步骤3】地图恢复：恢复原始表示（1为自由空间，0为障碍物）
        MapDilate = abs(1-InvertMapDilate)  # 恢复地图表示：膨胀后的自由空间地图
        
        # 【步骤4】掩码集成：结合MPT预测的可行区域
        if MapMask is None:
            # 无MPT引导：直接使用膨胀后的地图
            self.MaskMapDilate = MapDilate>0.5  # 二值化：>0.5为可通行区域
        else:
            # 有MPT引导：取膨胀地图与MPT掩码的交集
            self.MaskMapDilate = np.logical_and(MapDilate, MapMask)  # 逻辑与：同时满足无碰撞和MPT可行
            
    def isValid(self, state):
        """
        检查给定状态是否有效（无碰撞且在允许区域内）
        
        【核心功能】
        判断车辆在指定状态下是否安全可行，包括碰撞检测和边界检查。
        这是OMPL规划器调用的核心接口，决定了搜索空间的有效性。
        
        【检查流程】
        1. 状态提取：获取车辆的位置坐标
        2. 坐标转换：将几何坐标转换为像素坐标
        3. 边界检查：确保位置在地图范围内
        4. 碰撞检查：查询预处理的有效性表
        
        Args:
            state (ob.State): OMPL状态对象
                包含：车辆的位置(x,y)和朝向(θ)
                坐标系：几何坐标系，单位为米
                范围：应在地图边界内
        
        Returns:
            bool: 状态有效性
                True: 状态有效，车辆可以安全到达此位置
                False: 状态无效，存在碰撞或超出边界
        
        【实现细节】
        1. 位置提取：仅考虑位置(x,y)，忽略朝向θ
           - 原因：碰撞检测基于车辆中心点
           - 简化：假设车辆为圆形，朝向不影响碰撞
        
        2. 坐标转换：几何坐标→像素坐标
           - 目的：将连续坐标映射到离散网格
           - 方法：使用geom2pix函数进行转换
        
        3. 边界检查：防止数组越界访问
           - 检查：像素坐标是否在[0, size-1]范围内
           - 必要性：避免程序崩溃和无效查询
        
        4. 有效性查询：O(1)时间复杂度的表查找
           - 数据源：预处理的MaskMapDilate数组
           - 结果：综合了碰撞检测和MPT引导的结果
        
        【性能优化】
        - 预处理策略：初始化时完成所有复杂计算
        - 快速查询：运行时仅需简单的数组索引
        - 内存换时间：存储完整的有效性表
        """
        x, y = state.getX(), state.getY()  # 提取车辆位置：获取几何坐标(x,y)
        pix_dim = geom2pix([x, y], size=self.size)  # 坐标转换：几何坐标→像素坐标
        
        # 【边界检查】确保像素坐标在有效范围内
        if pix_dim[0] < 0 or pix_dim[0] >= self.size[0] or pix_dim[1] < 0 or pix_dim[1] >= self.size[1]:
            return False  # 超出边界：状态无效
        
        # 【有效性查询】查询预处理的有效性表
        return self.MaskMapDilate[pix_dim[1], pix_dim[0]]  # 返回该位置的有效性（True/False）

# def get_path_sst(start, goal, input_map, patch_map):
#     '''
#     Plan a path using SST, but invert the start and goal location.
#     :param start: The starting position on map co-ordinates.
#     :param goal: The goal position on the map co-ordinates.
#     :param input_map: The input map 
#     :param patch_map: The patch map
#     :returns 
#     '''
#     success, time, _, path = get_path(start, goal, input_map, patch_map, use_valid_sampler=True)
#     return path, time, [], success


def get_path(start, goal, input_map, patch_map, step_time=0.1, max_time=300, exp=False):
    """
    基于SST算法的车辆路径规划函数
    
    【核心功能】
    使用SST（Stable Sparse Tree）算法为车辆规划从起点到终点的可行路径。
    结合MPT预测的引导区域，实现高效的非完整约束路径规划。
    
    【算法特点】
    1. SST规划器：专门处理动力学约束的随机采样规划算法
    2. 非完整约束：考虑车辆运动学限制，不能任意方向移动
    3. MPT引导：利用学习预测缩小搜索空间，提高效率
    4. 探索-利用：可选的两阶段规划策略
    
    【规划流程】
    1. 状态设置：配置起点和终点状态
    2. 有效性检查：设置碰撞检测和MPT引导
    3. 动力学配置：设置车辆运动学模型
    4. 规划执行：运行SST算法搜索路径
    5. 结果提取：获取路径、时间和统计信息
    
    Args:
        start (array): SE(2)起点坐标 [x, y, θ]
            x, y: 起点位置坐标（米）
            θ: 起点朝向角（弧度）
            坐标系：几何坐标系，原点在左下角
        goal (array): SE(2)终点坐标 [x, y, θ]
            x, y: 终点位置坐标（米）
            θ: 终点朝向角（弧度）
            约束：必须在地图边界内
        input_map (np.array): 输入地图
            格式：2D数组，0表示障碍物，1表示自由空间
            用途：碰撞检测的基础地图
        patch_map (np.array): MPT预测的引导区域
            格式：与input_map相同尺寸的数组
            含义：1表示MPT认为可行的区域
            作用：引导搜索，提高规划效率
            可选：None表示不使用MPT引导
        step_time (float): 规划器时间步长，默认0.1秒
            作用：控制规划器每次迭代的时间限制
            影响：较小值提供更精细的时间控制
        max_time (float): 最大规划时间，默认300秒
            作用：防止规划器无限运行
            权衡：更长时间可能找到更好解，但计算开销大
        exp (bool): 是否使用探索-利用策略，默认False
            True: 先在MPT引导区域搜索，再扩展到全空间
            False: 直接在全空间搜索
            优势：可能在复杂环境中提高成功率
    
    Returns:
        tuple: (path, time, numVertices, success)
            path (np.array): 规划的路径点序列，形状为(N, 3)
                每行：[x, y, θ] 表示路径上的一个状态
                单位：位置为米，角度为弧度
                空列表：规划失败时返回[]
            time (float): 规划耗时（秒）
                包含：所有规划阶段的总时间
                用途：性能评估和算法对比
            numVertices (int): 搜索树中的节点数量
                含义：反映搜索的复杂度
                用途：算法效率分析
            success (bool): 规划是否成功
                True: 找到可行路径
                False: 在时间限制内未找到路径
    
    【技术创新】
    1. 混合引导策略：结合学习预测和传统规划
    2. 动态有效性检查：根据规划阶段切换检查器
    3. 时间管理：精确的时间控制和超时处理
    4. 统计收集：完整的规划性能指标
    """
    # 【采样策略说明】
    # 尝试过重要性采样，但相比拒绝采样没有显著改进
    # 当前使用标准的均匀随机采样策略

    # 【步骤1】起点状态配置
    StartState = ob.State(space)  # 创建起点状态对象
    StartState().setX(start[0])   # 设置起点X坐标
    StartState().setY(start[1])   # 设置起点Y坐标
    StartState().setYaw(start[2]) # 设置起点朝向角

    # 【步骤2】终点状态配置
    GoalState = ob.State(space)   # 创建终点状态对象
    GoalState().setX(goal[0])     # 设置终点X坐标
    GoalState().setY(goal[1])     # 设置终点Y坐标
    GoalState().setYaw(goal[2])   # 设置终点朝向角

    # 【步骤3】规划器初始化
    success = False  # 初始化成功标志
    ss = oc.SimpleSetup(cspace)  # 创建控制规划设置对象
    
    # 【步骤4】有效性检查器配置
    # 创建两个检查器：一个带MPT引导，一个不带引导
    ValidityCheckerObj = ValidityChecker(si, input_map, patch_map)  # MPT引导的检查器
    NewValidityCheckerObj = ValidityChecker(si, input_map)  # 标准检查器（无MPT引导）

    ss.setStateValidityChecker(ValidityCheckerObj)  # 设置初始检查器（带MPT引导）
    
    # 【步骤5】起点终点设置
    ss.setStartAndGoalStates(StartState, GoalState, 1.0)  # 设置起点和终点，容差为1.0米

    # 【步骤6】动力学模型配置
    ode = oc.ODE(kinematicCarODE)  # 创建ODE对象，使用车辆运动学方程
    odeSolver = oc.ODEBasicSolver(ss.getSpaceInformation(), ode)  # 创建ODE求解器
    propagator = oc.ODESolver.getStatePropagator(odeSolver)  # 获取状态传播器
    ss.setStatePropagator(propagator)  # 设置状态传播器
    ss.getSpaceInformation().setPropagationStepSize(0.1)  # 设置传播步长为0.1秒
    # ss.getSpaceInformation().setMinMaxControlDuration(1, 10)  # 可选：设置控制持续时间范围

    # 【步骤7】规划算法配置
    planner = oc.SST(ss.getSpaceInformation())  # 创建SST规划器
    ss.setPlanner(planner)  # 设置规划器

    # 【步骤8】规划执行
    time = step_time  # 初始化计时器
    
    # 【探索-利用策略】（可选）
    if exp:
        # 第一阶段：在MPT引导区域内快速搜索
        solved = ss.solve(time)  # 开始规划
        while not ss.haveExactSolutionPath():  # 如果没有找到精确解
            solved = ss.solve(step_time)  # 继续规划
            time += step_time  # 累计时间
            if time > 2:  # 限制第一阶段时间为2秒
                break
        # 切换到无MPT引导的检查器，扩大搜索空间
        ss.setStateValidityChecker(NewValidityCheckerObj)

    # 【主要规划阶段】
    solved = ss.solve(step_time)  # 开始或继续规划
    while not ss.haveExactSolutionPath():  # 循环直到找到解或超时
        solved = ss.solve(step_time)  # 继续规划
        time += step_time  # 累计时间
        if time > max_time:  # 检查是否超过最大时间限制
            break

    # 【步骤9】结果处理
    if ss.haveExactSolutionPath():  # 如果找到了解
        success = True  # 标记成功
        print("Found Solution")  # 输出成功信息
        
        # 提取路径：将OMPL路径转换为numpy数组
        path = np.array([[ss.getSolutionPath().getState(i).getX(), 
                         ss.getSolutionPath().getState(i).getY(), 
                         ss.getSolutionPath().getState(i).getYaw()]
                        for i in range(ss.getSolutionPath().getStateCount())])
        
        # 计算路径质量：总的欧几里得距离
        path_quality = 0
        for i in range(len(path)-1):
            path_quality += np.linalg.norm(path[i+1, :2]-path[i, :2])  # 累计相邻点间距离
    else:
        # 规划失败
        success = False  # 标记失败
        path_quality = np.inf  # 路径质量设为无穷大
        path = []  # 空路径

    # 【步骤10】统计信息收集
    plannerData = ob.PlannerData(si)  # 创建规划数据对象
    planner.getPlannerData(plannerData)  # 获取规划器数据
    numVertices = plannerData.numVertices()  # 获取搜索树节点数量
    
    return path, time, numVertices, success  # 返回路径、时间、节点数和成功标志

# 【计算设备配置】
# 自动检测并选择最优的计算设备：优先使用GPU加速模型推理
device='cuda' if torch.cuda.is_available() else 'cpu'  # GPU可用时使用CUDA，否则使用CPU


def get_patch(model, start_pos, goal_pos, input_map):
    """
    基于MPT模型生成路径规划引导区域
    
    【核心功能】
    使用训练好的Transformer模型预测路径规划的关键区域，生成用于引导
    传统规划算法的patch map，提高规划效率和成功率。
    
    【工作流程】
    1. 输入编码：将地图、起点、终点编码为模型输入
    2. 模型推理：使用MPT预测每个位置的重要性
    3. 锚点识别：找出模型认为重要的位置
    4. 区域生成：围绕锚点生成感受野大小的patch区域
    
    Args:
        model (torch.nn.Module): 训练好的MPT Transformer模型
            输入：编码后的地图和起终点信息
            输出：每个位置的二分类概率（重要/不重要）
        start_pos (tuple): 起点像素坐标 (x, y)
            格式：像素坐标系，原点在左上角
            用途：为模型提供起点信息
        goal_pos (tuple): 终点像素坐标 (x, y)
            格式：像素坐标系，原点在左上角
            用途：为模型提供终点信息
        input_map (np.array): 输入地图
            格式：2D数组，0表示障碍物，1表示自由空间
            尺寸：通常为480x480像素
    
    Returns:
        np.array: patch map引导区域
            格式：与input_map相同尺寸的二值数组
            含义：1表示MPT认为重要的区域，0表示不重要
            用途：引导传统规划算法的搜索空间
    
    【技术细节】
    1. 模型输入编码：
       - 使用get_encoder_input函数处理输入
       - 将地图、起点、终点信息融合
       - 转换为模型可接受的张量格式
    
    2. 模型推理：
       - 前向传播获得预测结果
       - 输出形状：(batch_size, num_tokens, num_classes)
       - 每个token对应地图上的一个区域
    
    3. 锚点识别：
       - 使用argmax获得每个token的预测类别
       - 筛选出预测为"重要"（类别1）的token
       - 通过hashTable将token索引映射到像素坐标
    
    4. Patch区域生成：
       - 以每个锚点为中心，生成receptive_field大小的正方形区域
       - 处理边界情况，确保不超出地图范围
       - 所有patch区域的并集构成最终的引导区域
    """
    # 【步骤1】输入编码：准备模型输入数据
    encoder_input = get_encoder_input(input_map, goal_pos, start_pos)  # 编码地图和起终点信息
    
    # 【步骤2】模型推理：获取MPT预测结果
    predVal = model(encoder_input[None,:].float().cuda())  # 模型前向传播，添加batch维度并移到GPU
    predClass = predVal[0, :, :].max(1)[1]  # 获取每个token的预测类别（argmax）

    # 【步骤3】概率计算和锚点识别
    predProb = F.softmax(predVal[0, :, :], dim=1)  # 计算softmax概率（用于分析，当前未使用）
    possAnchor = [hashTable[i] for i, label in enumerate(predClass) if label==1]  # 筛选重要锚点
    
    # 【锚点筛选逻辑】
    # - 遍历所有576个token（24x24网格）
    # - 选择预测类别为1（重要）的token
    # - 通过hashTable将token索引转换为像素坐标

    # 【步骤4】生成Patch Map：构建引导区域
    patch_map = np.zeros_like(input_map)  # 初始化patch map，与输入地图同尺寸
    map_size = input_map.shape  # 获取地图尺寸，用于边界检查
    
    # 遍历所有重要锚点，生成对应的patch区域
    for pos in possAnchor:
        # 计算patch区域的边界，以锚点为中心，receptive_field为边长
        goal_start_x = max(0, pos[0]- receptive_field//2)  # X方向起始位置，确保不小于0
        goal_start_y = max(0, pos[1]- receptive_field//2)  # Y方向起始位置，确保不小于0
        goal_end_x = min(map_size[0], pos[0]+ receptive_field//2)  # X方向结束位置，确保不超出地图
        goal_end_y = min(map_size[1], pos[1]+ receptive_field//2)  # Y方向结束位置，确保不超出地图
        
        # 在patch区域内设置为1，表示重要区域
        patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
    
    # 【结果说明】
    # patch_map: 最终的引导区域图
    # - 1.0: MPT认为路径可能经过的重要区域
    # - 0.0: MPT认为路径不太可能经过的区域
    # - 用途: 限制传统规划算法的搜索空间，提高效率
    
    return patch_map

# 【计算设备配置】
# 自动检测并选择最优的计算设备：优先使用GPU加速模型推理
device='cuda' if torch.cuda.is_available() else 'cpu'  # GPU可用时使用CUDA，否则使用CPU

if __name__=="__main__":
    """
    【主程序】车辆路径规划评估脚本
    
    【功能概述】
    批量评估MPT引导的车辆路径规划算法性能，支持多种评估模式：
    1. 纯SST规划（基线方法）
    2. MPT+SST混合规划（提出方法）
    3. 探索-利用策略的MPT+SST规划
    
    【评估流程】
    1. 参数解析：处理命令行参数
    2. 模型加载：加载训练好的MPT模型（如果需要）
    3. 批量测试：遍历测试环境和路径
    4. 性能统计：收集规划时间、成功率等指标
    5. 结果保存：保存评估结果供后续分析
    
    【命令行使用示例】
    # 纯SST评估
    python eval_model_car.py --modelFolder ./models --valDataFolder ./data --start 0 --numEnv 100 --use_sst
    
    # MPT+SST评估
    python eval_model_car.py --modelFolder ./models --valDataFolder ./data --start 0 --numEnv 100
    
    # 探索-利用模式
    python eval_model_car.py --modelFolder ./models --valDataFolder ./data --start 0 --numEnv 100 --explore
    """
    # 【步骤1】命令行参数配置
    parser = argparse.ArgumentParser(description='车辆模型路径规划评估脚本')
    parser.add_argument('--modelFolder', help='模型参数文件夹路径（包含model_params.json）', required=True)
    parser.add_argument('--valDataFolder', help='验证数据文件夹路径', required=True)
    parser.add_argument('--start', help='起始环境编号', required=True, type=int)
    parser.add_argument('--numEnv', help='测试环境数量', required=True, type=int)
    parser.add_argument('--numPaths', help='每个环境的起终点对数量', default=1, type=int)
    parser.add_argument('--use_sst', help='仅使用SST规划器（不使用MPT引导）', dest='use_sst', action='store_true')
    parser.add_argument('--explore', help='使用探索-利用策略', dest='explore', action='store_true')

    args = parser.parse_args()

    # 【步骤2】参数提取和验证
    modelFolder = args.modelFolder  # 模型文件夹路径
    modelFile = osp.join(modelFolder, f'model_params.json')  # 模型参数文件路径
    assert osp.isfile(modelFile), f"Cannot find the model_params.json file in {modelFolder}"

    start = args.start  # 起始环境编号
    use_sst = args.use_sst  # 是否仅使用SST
    valDataFolder = args.valDataFolder  # 验证数据路径

    # 【步骤3】模型加载（如果使用MPT引导）
    if not use_sst:
        # 加载模型参数配置
        model_param = json.load(open(modelFile))
        transformer = Models.Transformer(**model_param)  # 创建Transformer模型
        transformer.to(device)  # 移动到GPU（如果可用）

        receptive_field = 32  # 设置感受野大小
        
        # 加载预训练模型权重
        epoch = 69  # 使用第69轮的模型权重
        checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
        transformer.load_state_dict(checkpoint['state_dict'])

        # 设置为评估模式
        transformer.eval()  # 关闭dropout和batch normalization的训练模式
    
    # 【步骤4】初始化评估指标收集器
    pathSuccess = []  # 规划成功标志列表
    pathTime = []     # 规划时间列表
    pathVertices = [] # 搜索树节点数列表

    # 【步骤5】批量评估循环
    for env_num in range(start, start+args.numEnv):
        # 加载当前环境的地图
        temp_map = osp.join(valDataFolder, f'env{env_num:06d}/map_{env_num}.png')
        small_map = skimage.io.imread(temp_map, as_gray=True)  # 读取灰度地图

        # 遍历当前环境的所有路径任务
        for pathNum in range(args.numPaths):
            print(f"planning on env_{env_num} path_{pathNum}")  # 输出当前处理的任务
            
            # 加载路径数据
            pathFile = osp.join(valDataFolder, f'env{env_num:06d}/path_{pathNum}.p')
            data = pickle.load(open(pathFile, 'rb'))  # 加载pickle格式的路径数据
            path = data['path_interpolated']  # 获取插值后的参考路径

            # 仅处理有效的路径任务
            if data['success']:
                if not use_sst:
                    # 【MPT+SST混合规划模式】
                    
                    # 坐标转换：几何坐标→像素坐标
                    goal_pos = geom2pix(path[-1, :])   # 终点坐标转换
                    start_pos = geom2pix(path[0, :])   # 起点坐标转换

                    # MPT模型推理：生成引导区域
                    encoder_input = get_encoder_input(small_map, goal_pos, start_pos)  # 编码输入
                    with torch.no_grad():  # 禁用梯度计算，节省内存
                        predVal = transformer(encoder_input[None,:].float().cuda())  # 模型推理
                    predProb = F.softmax(predVal[0, :, :], dim=1)  # 计算概率分布

                    # 锚点识别和patch区域生成
                    predClass = predVal[0, :, :].max(1)[1]  # 获取预测类别
                    possAnchor = [hashTable[i] for i, label in enumerate(predClass) if label==1]  # 筛选重要锚点

                    # 生成patch map
                    patch_map = np.zeros_like(small_map)  # 初始化patch map
                    map_size = small_map.shape
                    for pos in possAnchor:
                        # 计算patch区域边界
                        goal_start_x = max(0, pos[0]- receptive_field//2)
                        goal_start_y = max(0, pos[1]- receptive_field//2)
                        goal_end_x = min(map_size[0], pos[0]+ receptive_field//2)
                        goal_end_y = min(map_size[1], pos[1]+ receptive_field//2)
                        patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
                    
                    # 执行MPT引导的路径规划
                    _, time, numVer, success = get_path(path[0, :], path[-1, :], small_map, 
                                                       patch_map.T, max_time=90, exp=args.explore)
                else:
                    # 【纯SST规划模式】（基线方法）
                    _, time, numVer, success = get_path(path[0, :], path[-1, :], small_map, 
                                                       patch_map=None, max_time=150)
                
                # 记录评估结果
                pathSuccess.append(success)    # 规划成功标志
                pathTime.append(time)          # 规划耗时
                pathVertices.append(numVer)    # 搜索树节点数
            else:
                # 处理无效的路径任务（参考路径规划失败的情况）
                # 注意：这里存在潜在的bug，time和numVer变量可能未定义
                pathSuccess.append(False)      # 标记为失败
                pathTime.append(0.0)           # 设置默认时间值
                pathVertices.append(0)         # 设置默认节点数
                    # np.save(f'{result_folder}/PathSuccess_{start}.npy', PathSuccess)
                    # np.save(f'{result_folder}/TimeSuccess_{start}.npy', TimeSuccess)
                    # np.save(f'{result_folder}/QualitySuccess{start}.npy', QualitySuccess)
    # pickle.dump(PathSuccess, open(osp.join(modelFolder, f'eval_unknown_plan_{start:06d}.p'), 'wb'))
    # 【步骤6】结果整理和保存
    pathData = {'Time': pathTime, 'Success': pathSuccess, 'Vertices': pathVertices}  # 整理评估数据
    
    # 根据评估模式确定输出文件名
    if use_sst:
        # 纯SST模式
        fileName = osp.join(modelFolder, f'eval_val_plan_sst_{start:06d}.p')
    else:
        if args.explore:
            # MPT+SST探索-利用模式
            fileName = osp.join(modelFolder, f'eval_val_plan_exp_mpt_sst_{start:06d}.p')
        else:
            # 标准MPT+SST模式
            fileName = osp.join(modelFolder, f'eval_val_plan_mpt_sst_{start:06d}.p')
    
    # 保存评估结果
    pickle.dump(pathData, open(fileName, 'wb'))  # 以pickle格式保存数据
    print(len(pathSuccess))  # 输出处理的路径总数
    
    # 【评估完成】
    # 输出文件包含三个关键指标：
    # - Time: 每个路径的规划时间（秒）
    # - Success: 每个路径的规划成功标志（布尔值）
    # - Vertices: 每个路径规划时搜索树的节点数量
    # 这些数据可用于后续的性能分析和算法对比
