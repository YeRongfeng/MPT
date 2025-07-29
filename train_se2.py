"""
train_se2.py - 在原有的训练脚本上拓展了对锚点朝向的处理，以适应SE(2)环境的训练需求。
"""

import numpy as np
import pickle

import torch
import torch.optim as optim

import json
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp

from transformer import Models, Optim
from dataLoader import PathSE2DataLoader, PaddedSequence, PathMixedDataLoader

from torch.utils.tensorboard import SummaryWriter


def focal_loss(predVals, trueLabels, gamma, eps=1e-8):
    """
    焦点损失(Focal Loss)计算函数
    
    【核心功能】
    实现论文https://arxiv.org/pdf/1708.02002.pdf中提出的焦点损失，
    该损失函数通过降低易分类样本的权重，增加难分类样本的权重，
    解决类别不平衡问题，提高模型对困难样本的关注度。
    
    【算法原理】
    焦点损失对标准交叉熵损失进行修改，添加调制因子(1-pt)^gamma：
    FL(pt) = -(1-pt)^gamma * log(pt)
    其中pt是模型对正确类别的预测概率，gamma是调节难易样本权重的超参数。
    
    Args:
        predVals (torch.Tensor): 模型输出的预测值，形状为(batch_size, num_classes)
            内容：最终线性层的原始输出(logits)
            范围：通常为未归一化的实数值
        trueLabels (torch.Tensor): 真实标签，形状为(batch_size,)
            内容：每个样本的类别索引
            范围：整数，从0到num_classes-1
        gamma (float): 焦点损失的调制因子
            作用：控制易分类样本的权重降低程度
            范围：通常为1-5，值越大对难分类样本的关注越多
        eps (float): 数值稳定性的小常数，默认1e-8
            作用：防止对数计算中的零值导致的数值不稳定
    
    Returns:
        torch.Tensor: 计算得到的焦点损失值
            形状：标量
            含义：整个批次的总损失
    
    【实现步骤】
    1. 对预测值应用softmax获取概率分布
    2. 将真实标签转换为one-hot编码
    3. 计算调制因子(1-pt)^gamma
    4. 应用对数损失并与调制因子相乘
    5. 对所有样本和类别求和得到总损失
    
    【应用场景】
    - 类别不平衡问题：当正负样本比例严重失衡时
    - 难样本学习：需要模型更关注难以分类的样本
    - 目标检测：处理前景-背景类别不平衡
    """
    input_soft = F.softmax(predVals, dim=1) + eps  # 对预测值应用softmax获取概率分布，加eps确保数值稳定性
    target_one_hot = torch.zeros((trueLabels.shape[0], 2), device=trueLabels.device)  # 创建空的one-hot编码张量
    target_one_hot.scatter_(1, trueLabels.unsqueeze(1), 1.0)  # 将真实标签转换为one-hot编码

    weight = torch.pow(-input_soft + 1., gamma)  # 计算调制因子(1-pt)^gamma，降低易分类样本的权重
    focal = -weight*torch.log(input_soft)  # 应用对数损失并与调制因子相乘
    loss = torch.sum(target_one_hot*focal, dim=1).sum()  # 对所有样本和类别求和得到总损失
    return loss

def cal_performance(predVals, anchorPoints, trueLabels, lengths, anchorThetas):
    """
    计算模型性能指标（支持SE2朝向损失）
    
    【新增】
    在原有的性能计算基础上，增加了对锚点朝向的处理，以适应SE(2)环境的训练需求。
    “maximizing cosine similarity measure for the orientation”
    最大化余弦相似度度量朝向，鼓励模型学习到更准确的朝向信息，也就是新的loss，我们称它为orientation loss。
    这个新的loss我们会在原有的交叉熵损失上加权求和。
    
    【核心功能】
    评估模型在给定批次数据上的性能，计算损失和正确预测的数量。
    针对变长序列数据，只考虑有效的锚点位置进行评估。
    
    【评估原理】
    对每个样本，提取对应锚点位置的预测值，与真实标签计算交叉熵损失，
    并统计预测正确的比例作为准确率指标。
    
    Args:
        predVals (list[torch.Tensor]): 模型预测值列表
            形状：每个元素为(seq_len, num_classes)
            内容：每个位置的类别预测分数
        anchorPoints (list[torch.Tensor]): 锚点索引列表
            形状：每个元素为(length,)
            内容：需要评估的位置索引
            作用：从所有位置中选择关键点进行评估
        trueLabels (list[torch.Tensor]): 真实标签列表
            形状：每个元素为(length,)
            内容：每个锚点位置的真实类别
        lengths (list[int]): 每个样本的有效长度
            作用：处理变长序列，只考虑有效部分
    
    Returns:
        tuple: (total_loss, n_correct)
            total_loss (torch.Tensor): 批次总损失
                形状：标量
                计算：所有样本损失之和
            n_correct (float): 平均正确预测数
                范围：0到批次大小
                计算：每个样本的正确率之和
    
    【实现步骤】
    1. 遍历批次中的每个样本
    2. 根据锚点索引提取对应位置的预测值
    3. 计算交叉熵损失
    4. 统计预测正确的位置数量
    5. 返回总损失和平均正确预测数
    
    【应用场景】
    - 模型训练：计算损失用于反向传播
    - 模型评估：计算准确率衡量性能
    - 变长序列处理：处理不同长度的路径序列
    """
    n_correct = 0  # 初始化正确预测计数器
    total_loss = 0  # 初始化总损失
    orientation_weight = 0.2  # 朝向损失权重，可根据实验调整
    for i, (predVal, anchorPoint, trueLabel, length, anchor_theta) in enumerate(zip(predVals, anchorPoints, trueLabels, lengths, anchorThetas)):
        predVal_cls = predVal.index_select(0, anchorPoint[:length])
        trueLabel = trueLabel[:length]
        loss_cls = F.cross_entropy(predVal_cls[:, :2], trueLabel)
        device = trueLabel.device if hasattr(trueLabel, 'device') else 'cpu'
        pos_mask = (trueLabel == 1)
        if pos_mask.sum() > 0:
            pred_theta = predVal_cls[pos_mask, 2]
            true_theta = anchor_theta[:length][pos_mask].to(device)
            cos_sim = torch.cos(pred_theta - true_theta)
            loss_orientation = 1.0 - cos_sim.mean()
        else:
            loss_orientation = torch.tensor(0.0, device=device)
        loss = loss_cls + orientation_weight * loss_orientation
        total_loss += loss
        classPred = predVal_cls[:, :2].max(1)[1]
        n_correct += classPred.eq(trueLabel).sum().item()/length
    return total_loss, n_correct

def train_epoch(model, trainingData, optimizer, device):
    """
    单轮训练函数
    
    【核心功能】
    对模型进行一个完整epoch的训练，包括前向传播、损失计算、反向传播和参数更新。
    使用tqdm显示训练进度，并返回整个epoch的损失和准确率统计。
    
    【训练流程】
    1. 设置模型为训练模式
    2. 遍历训练数据批次
    3. 执行前向传播计算预测值
    4. 计算损失和准确率
    5. 执行反向传播和参数更新
    6. 返回整个epoch的统计结果
    
    Args:
        model (nn.Module): 要训练的Transformer模型
            类型：Models.Transformer或nn.DataParallel包装的模型
        trainingData (DataLoader): 训练数据加载器
            内容：包含地图、锚点和标签的批次数据
            格式：字典形式的批次，包含'map'、'anchor'、'labels'和'length'键
        optimizer (Optim.ScheduledOptim): 优化器
            特点：包含学习率调度功能的优化器包装器
        device (torch.device): 计算设备
            选项：'cpu'或'cuda'设备对象
    
    Returns:
        tuple: (total_loss, total_n_correct)
            total_loss (float): 整个epoch的总损失
            total_n_correct (float): 整个epoch的正确预测总数
    
    【实现细节】
    - 使用tqdm显示进度条，mininterval=2设置更新间隔
    - 每个批次开始前清零梯度
    - 使用step_and_update_lr同时更新参数和学习率
    - 累加每个批次的损失和正确预测数
    
    【训练技巧】
    - 模型设置为train()模式，启用dropout和batch normalization
    - 使用device确保数据和模型在同一设备上
    - 优化器包含学习率调度，自动调整学习率
    """
    model.train()  # 设置模型为训练模式：启用dropout和batch normalization
    total_loss = 0  # 初始化总损失
    total_n_correct = 0  # 初始化总正确预测数
    # Train for a single epoch.
    for batch in tqdm(trainingData, mininterval=2):  # 遍历训练数据批次，使用tqdm显示进度
        
        optimizer.zero_grad()  # 清零梯度：避免梯度累积
        encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
        predVal = model(encoder_input)  # 前向传播：获取模型预测值

        # Calculate the cross-entropy loss
        loss, n_correct = cal_performance(  # 计算性能指标：损失和正确预测数
            predVal, batch['anchor'].to(device),  # 预测值和锚点索引
            batch['labels'].to(device),  # 真实标签
            batch['length'].to(device)  # 序列有效长度
        )
        loss.backward()  # 反向传播：计算梯度
        optimizer.step_and_update_lr()  # 参数更新：同时更新模型参数和学习率
        total_loss += loss.item()  # 累加批次损失
        total_n_correct += n_correct  # 累加批次正确预测数
    return total_loss, total_n_correct  # 返回整个epoch的统计结果


def eval_epoch(model, validationData, device):
    """
    单轮评估函数
    
    【核心功能】
    在验证数据集上评估模型性能，计算损失和准确率指标。
    使用torch.no_grad()确保不计算梯度，提高评估效率和内存使用。
    
    【评估流程】
    1. 设置模型为评估模式
    2. 禁用梯度计算
    3. 遍历验证数据批次
    4. 执行前向传播获取预测值
    5. 计算损失和准确率
    6. 返回整个验证集的统计结果
    
    Args:
        model (nn.Module): 要评估的Transformer模型
            类型：Models.Transformer或nn.DataParallel包装的模型
        validationData (DataLoader): 验证数据加载器
            内容：包含地图、锚点和标签的批次数据
        device (torch.device): 计算设备
            选项：'cpu'或'cuda'设备对象
    
    Returns:
        tuple: (total_loss, total_n_correct)
            total_loss (float): 验证集总损失
            total_n_correct (float): 验证集正确预测总数
    
    【实现细节】
    - 使用model.eval()设置模型为评估模式
    - 使用torch.no_grad()上下文管理器禁用梯度计算
    - 使用tqdm显示评估进度
    - 累加每个批次的损失和正确预测数
    
    【评估技巧】
    - 评估时禁用dropout和batch normalization的训练行为
    - 不计算梯度可以节省内存并加速评估过程
    - 使用与训练相同的性能计算函数确保一致性
    """

    model.eval()  # 设置模型为评估模式：禁用dropout和batch normalization的训练行为
    total_loss = 0.0  # 初始化总损失
    total_n_correct = 0.0  # 初始化总正确预测数
    with torch.no_grad():  # 禁用梯度计算：节省内存并加速评估
        for batch in tqdm(validationData, mininterval=2):  # 遍历验证数据批次，使用tqdm显示进度

            encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
            predVal = model(encoder_input)  # 前向传播：获取模型预测值

            loss, n_correct = cal_performance(  # 计算性能指标：损失和正确预测数
                predVal, 
                batch['anchor'].to(device),  # 锚点索引
                batch['labels'].to(device),  # 真实标签
                batch['length'].to(device)  # 序列有效长度
            )

            total_loss += loss.item()  # 累加批次损失
            total_n_correct += n_correct  # 累加批次正确预测数
    return total_loss, total_n_correct  # 返回整个验证集的统计结果


def check_data_folders(folder):
    """
    检查数据文件夹结构
    
    【核心功能】
    验证指定文件夹是否包含训练和验证所需的子文件夹结构。
    确保数据组织符合预期格式，避免训练过程中因数据结构问题导致的错误。
    
    【检查内容】
    文件夹必须包含两个子文件夹：
    1. 'train'：包含训练数据
    2. 'val'：包含验证数据
    
    Args:
        folder (str): 要检查的数据文件夹路径
            格式：字符串路径，可以是相对或绝对路径
            要求：必须是一个有效的目录
    
    Raises:
        AssertionError: 当找不到'train'或'val'子文件夹时抛出
            错误信息：指明缺少哪个子文件夹
    
    【使用场景】
    - 训练前的数据准备检查
    - 确保数据加载不会因文件夹结构问题失败
    - 提供清晰的错误信息，便于用户排查数据组织问题
    """
    assert osp.isdir(osp.join(folder, 'train')), "Cannot find trainining data"  # 检查train子文件夹是否存在
    assert osp.isdir(osp.join(folder, 'val')), "Cannot find validation data"  # 检查val子文件夹是否存在

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--batchSize', help="Batch size per GPU", required=True, type=int)  # 添加批次大小参数
    parser.add_argument('--mazeDir', help="Directory with training and validation data for Maze", default=None)  # 添加迷宫数据目录参数
    parser.add_argument('--forestDir', help="Directory with training and validation data for Random Forest", default=None)  # 添加随机森林数据目录参数
    parser.add_argument('--fileDir', help="Directory to save training Data")  # 添加训练数据保存目录参数
    args = parser.parse_args()  # 解析命令行参数

    maze=False  # 初始化迷宫环境标志
    if args.mazeDir is not None:  # 检查是否提供了迷宫数据目录
        check_data_folders(args.mazeDir)  # 检查迷宫数据文件夹结构
        maze=True  # 设置迷宫环境标志为True
    forest=False  # 初始化随机森林环境标志
    if args.forestDir is not None:  # 检查是否提供了随机森林数据目录
        check_data_folders(args.forestDir)  # 检查随机森林数据文件夹结构
        forest=True  # 设置随机森林环境标志为True

    assert forest or maze, "Need to provide data folder for atleast one kind of environment"  # 确保至少提供了一种环境的数据
    dataFolder = args.mazeDir if not(maze and forest) and maze else args.forestDir  # 确定数据文件夹：如果只有一种环境，使用该环境的文件夹；如果两种都有，默认使用随机森林文件夹

    print(f"Using data from {dataFolder}")  # 打印使用的数据文件夹路径

    batch_size = args.batchSize  # 获取每个GPU的批次大小
    device = 'cpu'  # 默认使用CPU设备
    if torch.cuda.is_available():  # 检查是否有可用的CUDA设备
        print("Using GPU....")  # 打印使用GPU的信息
        device = torch.device('cuda')  # 设置设备为CUDA

    if torch.cuda.device_count() > 1:  # 检查是否有多个GPU
        batch_size = batch_size * torch.cuda.device_count()  # 根据GPU数量调整总批次大小
    print(f"Total batch size : {batch_size}")  # 打印总批次大小

    torch_seed = np.random.randint(low=0, high=1000)  # 生成随机种子
    torch.manual_seed(torch_seed)  # 设置PyTorch随机种子，确保结果可复现
    
    model_args = dict(  # 定义模型参数字典
        n_layers=6,  # Transformer编码器层数：6层
        n_heads=3,  # 多头注意力的头数：3个头
        d_k=512,  # Key向量的维度：512
        d_v=256,  # Value向量的维度：256
        d_model=512,  # 模型的主要特征维度：512
        d_inner=1024,  # 前馈网络的隐藏层维度：1024
        pad_idx=None,  # 填充标记的索引：无
        n_position=40*40,  # 支持的最大位置数：1600(40×40)
        dropout=0.1,  # Dropout概率：0.1
        train_shape=[24, 24],  # 训练时的地图形状：24×24
    )
    
    transformer = Models.Transformer(**model_args)  # 使用参数字典初始化Transformer模型

    if torch.cuda.device_count() > 1:  # 检查是否有多个GPU
        print("Using ", torch.cuda.device_count(), "GPUs")  # 打印使用的GPU数量
        transformer = nn.DataParallel(transformer)  # 使用DataParallel包装模型，实现数据并行
    transformer.to(device=device)  # 将模型移动到指定设备(CPU或GPU)

    # Define the optimizer
    # TODO: What does these parameters do ???
    optimizer = Optim.ScheduledOptim(  # 创建带有学习率调度的优化器
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),  # Adam优化器：动量参数(0.9, 0.98)，数值稳定性参数1e-9
        lr_mul = 0.5,  # 学习率乘数：0.5
        d_model = 256,  # 模型维度：用于学习率计算
        n_warmup_steps = 3200  # 预热步数：3200步
    )

    # Training with Mixed samples
    if maze and forest:  # 如果同时有迷宫和随机森林环境
        from toolz.itertoolz import partition  # 导入分区工具
        trainDataset= PathMixedDataLoader(  # 创建混合数据加载器
            envListForest=list(range(10)),  # 随机森林环境列表：0-9
            dataFolderForest=osp.join(args.forestDir, 'train'),  # 随机森林训练数据文件夹
            envListMaze=list(range(10)),  # 迷宫环境列表：0-9
            dataFolderMaze=osp.join(args.mazeDir, 'train')  # 迷宫训练数据文件夹
        )
        allTrainingData = trainDataset.indexDictForest + trainDataset.indexDictMaze  # 合并两种环境的训练数据索引
        batch_sampler_train = list(partition(batch_size, allTrainingData))  # 创建批次采样器：将数据分成固定大小的批次
        trainingData = DataLoader(trainDataset, num_workers=15, batch_sampler=batch_sampler_train, collate_fn=PaddedSequence)  # 创建训练数据加载器：15个工作线程，使用批次采样器，使用PaddedSequence处理变长序列

        valDataset = PathMixedDataLoader(  # 创建混合验证数据加载器
            envListForest=list(range(10)),  # 随机森林环境列表：0-9
            dataFolderForest=osp.join(args.forestDir, 'val'),  # 随机森林验证数据文件夹
            envListMaze=list(range(10)),  # 迷宫环境列表：0-9
            dataFolderMaze=osp.join(args.mazeDir, 'val')  # 迷宫验证数据文件夹
        )
        allValData = valDataset.indexDictForest+valDataset.indexDictMaze  # 合并两种环境的验证数据索引
        batch_sampler_val = list(partition(batch_size, allValData))  # 创建验证批次采样器
        validationData = DataLoader(valDataset, num_workers=5, batch_sampler=batch_sampler_val, collate_fn=PaddedSequence)  # 创建验证数据加载器：5个工作线程
    else:  # 如果只有一种环境      
        trainDataset = PathSE2DataLoader(  # 创建单一环境训练数据加载器（SE2）
            env_list=list(range(500)),
            dataFolder=osp.join(dataFolder, 'train')
        )
        trainingData = DataLoader(trainDataset, num_workers=15, collate_fn=PaddedSequence, batch_size=batch_size)

        # Validation Data
        valDataset = PathSE2DataLoader(
            env_list=list(range(500)),
            dataFolder=osp.join(dataFolder, 'val')
        )
        validationData = DataLoader(valDataset, num_workers=5, collate_fn=PaddedSequence, batch_size=batch_size)

    # Increase number of epochs.
    n_epochs = 70  # 设置训练轮数：70轮
    results = {}  # 初始化结果字典
    train_loss = []  # 初始化训练损失列表
    val_loss = []  # 初始化验证损失列表
    train_n_correct_list = []  # 初始化训练正确预测列表
    val_n_correct_list = []  # 初始化验证正确预测列表
    trainDataFolder = args.fileDir  # 获取训练数据保存目录
    # Save the model parameters as .json file
    json.dump(  # 保存模型参数为JSON文件
        model_args,  # 模型参数字典
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),  # 打开文件用于写入
        sort_keys=True,  # 按键排序
        indent=4  # 缩进4个空格，美化输出
    )
    writer = SummaryWriter(log_dir=trainDataFolder)  # 创建TensorBoard摘要写入器，用于记录训练过程
    for n in range(n_epochs):  # 遍历训练轮数
        train_total_loss, train_n_correct = train_epoch(transformer, trainingData, optimizer, device)  # 执行一轮训练
        val_total_loss, val_n_correct = eval_epoch(transformer, validationData, device)  # 执行一轮验证
        print(f"Epoch {n} Loss: {train_total_loss}")  # 打印训练损失
        print(f"Epoch {n} Loss: {val_total_loss}")  # 打印验证损失
        print(f"Epoch {n} Accuracy {val_n_correct/len(valDataset)}")  # 打印验证准确率

        # Log data.
        train_loss.append(train_total_loss)  # 记录训练损失
        val_loss.append(val_total_loss)  # 记录验证损失
        train_n_correct_list.append(train_n_correct)  # 记录训练正确预测数
        val_n_correct_list.append(val_n_correct)  # 记录验证正确预测数

        if (n+1)%5==0:  # 每5轮保存一次模型
            if isinstance(transformer, nn.DataParallel):  # 检查模型是否使用DataParallel包装
                state_dict = transformer.module.state_dict()  # 获取原始模型的状态字典
            else:
                state_dict = transformer.state_dict()  # 直接获取模型状态字典
            states = {  # 创建保存状态字典
                'state_dict': state_dict,  # 模型参数
                'optimizer': optimizer._optimizer.state_dict(),  # 优化器状态
                'torch_seed': torch_seed  # 随机种子
            }
            torch.save(states, osp.join(trainDataFolder, 'model_epoch_{}.pkl'.format(n)))  # 保存模型检查点：包含模型参数、优化器状态和随机种子
        
        pickle.dump(  # 保存训练进度数据
            {
                'trainLoss': train_loss,  # 训练损失历史
                'valLoss':val_loss,  # 验证损失历史
                'trainNCorrect':train_n_correct_list,  # 训练正确预测历史
                'valNCorrect':val_n_correct_list  # 验证正确预测历史
            }, 
            open(osp.join(trainDataFolder, 'progress.pkl'), 'wb')  # 以二进制写入模式保存到progress.pkl文件
            )
        writer.add_scalar('Loss/train', train_total_loss, n)  # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/test', val_total_loss, n)  # 记录验证损失到TensorBoard
        writer.add_scalar('Accuracy/train', train_n_correct/len(trainDataset), n)  # 记录训练准确率到TensorBoard
        writer.add_scalar('Accuracy/test', val_n_correct/len(valDataset), n)  # 记录验证准确率到TensorBoard
