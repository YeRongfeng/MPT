"""
train_uneven.py - 训练不平坦地面路径预测模型
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
from dataLoader_uneven import UnevenPathDataLoader, PaddedSequence
from dataLoader_uneven import hashTable, receptive_field

from torch.utils.tensorboard import SummaryWriter

def cal_performance(predVals, anchorPoints, trueLabels, trajectory):
    """
    计算模型性能指标
    predVals: [batch_size, num_tokens, output_dim] - 模型预测：144个锚点位置，每个位置10个时间步的预测
    anchorPoints: [batch_size, num_layers, max_anchors] - 锚点索引
    trueLabels: [batch_size, num_layers, max_anchors] - 真实标签
    """
    n_correct = 0  # 初始化正确预测计数器
    total_loss = 0.0  # 初始化总损失为浮点数
    total_samples = 0  # 初始化总样本数
    batch_size, num_tokens, output_dim = predVals.shape
    
    loss_count = 0  # 计算实际损失的数量
    
    # 各损失的权重
    loss_weights = {
        'classification': 1.0,  # 分类损失权重(L_ce)
        'regression': 10e-3,      # 回归损失权重(L_mse)
    }

    # 用于统计标签分布
    total_positive = 0
    total_negative = 0
    correct_positive = 0
    correct_negative = 0
    
    for i in range(batch_size):
        anchorPoint = anchorPoints[i]  # [num_layers, max_anchors]
        trueLabel = trueLabels[i]  # [num_layers, max_anchors]
        
        # 损失1：锚点分类交叉熵损失(L_ce)
        # 对每个输出维度(时间步)分别计算损失和准确率
        for step in range(min(output_dim, anchorPoint.shape[0])):  # 确保不超出标签维度
            # 获取当前时间步的预测 [num_tokens] - 所有锚点位置在这个时间步的预测
            step_pred = predVals[i, :, step]  # [num_tokens] - 第step个时间步的预测
            
            # 获取当前时间步对应的锚点和标签
            anchor_step = anchorPoint[step]  # [max_anchors]
            label_step = trueLabel[step]     # [max_anchors]
            
            # 过滤掉填充值(-1)
            valid_mask = (anchor_step != -1) & (label_step != -1)
            valid_anchors = anchor_step[valid_mask]
            valid_labels = label_step[valid_mask]
            
            if len(valid_anchors) == 0:  # 如果没有有效的锚点，跳过
                continue
            
            # 检查锚点索引是否在有效范围内
            max_anchor_idx = valid_anchors.max().item()
            if max_anchor_idx >= num_tokens:
                print(f"Warning: anchor index {max_anchor_idx} >= num_tokens {num_tokens}, skipping")
                continue
            
            # 创建全图标签：默认所有位置都是负样本(0)
            full_labels = torch.zeros(num_tokens, device=predVals.device)
            
            # 将有效锚点位置设置为对应的真实标签
            full_labels[valid_anchors] = valid_labels.float()
            
            # 对所有位置计算BCE loss，这样背景区域会被推向低概率
            loss = F.binary_cross_entropy_with_logits(step_pred, full_labels)
            total_loss += loss * loss_weights['classification']  # 分类损失(L_ce)
            loss_count += 1
            
            # 计算准确率 - 只在锚点位置计算准确率
            selected_preds = step_pred.index_select(0, valid_anchors)
            selected_probs = torch.sigmoid(selected_preds)
            classPred = (selected_probs > 0.5).long()  # 将概率转换为预测类别
            
            # 统计正确预测数（只统计锚点位置）
            correct_predictions = classPred.eq(valid_labels.long()).sum().item()
            n_correct += correct_predictions
            total_samples += len(valid_labels)
            
            # 统计标签分布（只统计锚点位置）
            positive_labels = (valid_labels == 1).sum().item()
            negative_labels = (valid_labels == 0).sum().item()
            total_positive += positive_labels
            total_negative += negative_labels
            
            # 统计各类别的正确预测（只统计锚点位置）
            correct_positive += ((classPred == 1) & (valid_labels == 1)).sum().item()
            correct_negative += ((classPred == 0) & (valid_labels == 0)).sum().item()
            
        # 损失2：轨迹坐标回归损失(L_mse) - 使用张量并行计算
        # 确定处理的时间步数
        steps_to_process = min(output_dim, anchorPoint.shape[0])
        if steps_to_process > 0:
            # 将hashTable转换为张量以便并行计算
            hash_table_tensor = torch.tensor(hashTable, device=predVals.device)  # [num_tokens, 2]
            
            # 对所有时间步进行softmax归一化
            pred_probs = F.softmax(predVals[i, :, :steps_to_process], dim=0)  # [num_tokens, steps_to_process]
            
            # 计算所有时间步的加权坐标
            # 扩展hashTable以计算所有时间步 [num_tokens, 2, steps_to_process]
            expanded_hash = hash_table_tensor.unsqueeze(-1).expand(-1, -1, steps_to_process)
            
            # 扩展概率以匹配哈希表维度 [num_tokens, steps_to_process] -> [num_tokens, 1, steps_to_process]
            expanded_probs = pred_probs.unsqueeze(1)
            
            # 乘法和求和得到加权坐标 [2, steps_to_process]
            weighted_coords = torch.sum(expanded_hash * expanded_probs, dim=0)
            
            # 将像素坐标映射回实际坐标系
            weighted_coords = -5.0 + weighted_coords * 0.1
            
            # 提取真实轨迹坐标 [steps_to_process, 2]
            true_coords = trajectory[i, :steps_to_process, :2]
            
            # 计算所有时间步的MSE损失
            mse_loss = F.mse_loss(weighted_coords.t(), true_coords)
            total_loss += mse_loss * loss_weights['regression'] * steps_to_process  # 回归损失(L_mse)
            loss_count += steps_to_process
    
    # 确保返回张量类型的损失
    if loss_count == 0:
        # 如果没有计算任何损失，返回一个零损失张量
        total_loss = torch.tensor(0.0, requires_grad=True, device=predVals.device)
    
    # 返回总正确数而不是平均准确率，让调用者计算最终准确率
    return total_loss, n_correct, total_samples, (total_positive, total_negative, correct_positive, correct_negative)

def train_epoch(model, trainingData, optimizer, device):
    """
    单轮训练函数
    """
    model.train()  # 设置模型为训练模式：启用dropout和batch normalization
    total_loss = 0  # 初始化总损失
    total_n_correct = 0  # 初始化总正确预测数
    total_samples = 0  # 初始化总样本数
    epoch_stats = [0, 0, 0, 0]  # [total_positive, total_negative, correct_positive, correct_negative]
    
    # Train for a single epoch.
    for batch in tqdm(trainingData, mininterval=2):  # 遍历训练数据批次，使用tqdm显示进度
        
        optimizer.zero_grad()  # 清零梯度：避免梯度累积     
        encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
        predVal = model(encoder_input)  # 前向传播：获取模型预测值

        # 正确处理锚点和标签，保持对应关系
        loss, n_correct, n_samples, batch_stats = cal_performance(
            predVal, 
            batch['anchor'].to(device),
            batch['labels'].to(device),
            batch['trajectory'].to(device)  # 轨迹点：[N, 3]
        )
        loss.backward()  # 反向传播：计算梯度
        optimizer.step_and_update_lr()  # 参数更新：同时更新模型参数和学习率
        total_loss += loss.item()  # 累加批次损失
        total_n_correct += n_correct  # 累加批次正确预测数
        total_samples += n_samples  # 累加总样本数
        
        # 累加统计信息
        for i in range(4):
            epoch_stats[i] += batch_stats[i]
    
    return total_loss, total_n_correct, total_samples, epoch_stats  # 返回整个epoch的统计结果


def eval_epoch(model, validationData, device):
    """
    单轮评估函数
    """

    model.eval()  # 设置模型为评估模式：禁用dropout和batch normalization的训练行为
    total_loss = 0.0  # 初始化总损失
    total_n_correct = 0.0  # 初始化总正确预测数
    total_samples = 0  # 初始化总样本数
    epoch_stats = [0, 0, 0, 0]  # [total_positive, total_negative, correct_positive, correct_negative]
    
    with torch.no_grad():  # 禁用梯度计算：节省内存并加速评估
        for batch in tqdm(validationData, mininterval=2):  # 遍历验证数据批次，使用tqdm显示进度

            encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
            predVal = model(encoder_input)  # 前向传播：获取模型预测值

            # 正确处理锚点和标签，保持对应关系
            loss, n_correct, n_samples, batch_stats = cal_performance(
                predVal,
                batch['anchor'].to(device),
                batch['labels'].to(device),
                batch['trajectory'].to(device)  # 轨迹点：[N, 3]
            )

            total_loss += loss.item()  # 累加批次损失
            total_n_correct += n_correct  # 累加批次正确预测数
            total_samples += n_samples  # 累加总样本数
            
            # 累加统计信息
            for i in range(4):
                epoch_stats[i] += batch_stats[i]
                
    return total_loss, total_n_correct, total_samples, epoch_stats  # 返回整个验证集的统计结果


def check_data_folders(folder):
    """
    检查数据文件夹结构
    """
    assert osp.isdir(osp.join(folder, 'train')), "Cannot find training data"  # 检查train子文件夹是否存在
    assert osp.isdir(osp.join(folder, 'val')), "Cannot find validation data"  # 检查val子文件夹是否存在

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--batchSize', help="Batch size per GPU", required=True, type=int)  # 添加批次大小参数
    parser.add_argument('--env_list', help="Directory with training and validation data for Maze", default=None) 
    parser.add_argument('--dataFolder', help="Directory with training and validation data for Maze", default=None)  # 添加数据文件夹参数
    parser.add_argument('--fileDir', help="Directory to save training Data")  # 添加训练数据保存目录参数
    args = parser.parse_args()  # 解析命令行参数

    map_load = False  # 是否加载地图数据
    dataFolder = args.dataFolder  # 确定数据文件夹：使用提供的环境数据文件夹
    if not osp.isdir(dataFolder):  # 检查数据文件夹是否存在
        raise ValueError("Please provide a valid data folder")  # 如果不存在，抛出错误
    
    assert args.env_list is not None, "Please provide environment list"  # 确保提供了环境列表
    env_list = args.env_list.split(',')  # 将环境列表字符串分割成列表
    
    check_data_folders(dataFolder) # 检查数据文件夹结构
    map_load = True  # 设置加载地图数据标志为True

    assert map_load, "Need to provide data folder for atleast one kind of environment"  # 确保至少提供了一种环境的数据
    if not osp.isdir(args.fileDir):  # 检查训练数据保存目录是否存在
        raise ValueError("Please provide a valid file directory to save training data")  # 如果不存在，抛出错误

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
    
    model_args = dict(        # 定义模型参数字典
        n_layers=6,           # Transformer编码器层数：6层
        n_heads=3,            # 多头注意力的头数：3个头
        d_k=512,              # Key向量的维度：512
        d_v=256,              # Value向量的维度：256
        d_model=512,          # 模型的主要特征维度：512
        d_inner=1024,         # 前馈网络的隐藏层维度：1024
        pad_idx=None,         # 填充标记的索引：无
        n_position=15*15,     # 支持的最大位置数：225(15×15)
        dropout=0.1,          # Dropout概率：0.1
        train_shape=[12, 12], # 训练时的地图形状：12×12
        output_dim=10,        # 输出维度：10
    )

    transformer = Models.UnevenTransformer(**model_args)  # 使用参数字典初始化Transformer模型

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

    # Training Data   
    trainDataset = UnevenPathDataLoader(
        env_list=env_list,  # 使用提供的环境列表参数
        dataFolder=osp.join(dataFolder, 'train')
    )
    trainingData = DataLoader(trainDataset, num_workers=15, collate_fn=PaddedSequence, batch_size=batch_size)

    # Validation Data
    valDataset = UnevenPathDataLoader(
        env_list=env_list,  # 使用提供的环境列表参数
        dataFolder=osp.join(dataFolder, 'val')
    )
    validationData = DataLoader(valDataset, num_workers=5, collate_fn=PaddedSequence, batch_size=batch_size)

    # Increase number of epochs.
    n_epochs = 100  # 设置训练轮数：100轮
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
        train_total_loss, train_n_correct, train_samples, train_stats = train_epoch(transformer, trainingData, optimizer, device)  # 执行一轮训练
        val_total_loss, val_n_correct, val_samples, val_stats = eval_epoch(transformer, validationData, device)  # 执行一轮验证
        
        # 计算训练和验证准确率
        train_accuracy = train_n_correct / train_samples if train_samples > 0 else 0.0
        val_accuracy = val_n_correct / val_samples if val_samples > 0 else 0.0
        
        print(f"Epoch {n} Train Loss: {train_total_loss:.4f}")  # 打印训练损失
        print(f"Epoch {n} Val Loss: {val_total_loss:.4f}")  # 打印验证损失
        print(f"Epoch {n} Train Accuracy: {train_accuracy:.4f} ({train_n_correct}/{train_samples})")  # 打印训练准确率
        print(f"Epoch {n} Val Accuracy: {val_accuracy:.4f} ({val_n_correct}/{val_samples})")  # 打印验证准确率
        
        # 打印详细的分类统计
        total_pos, total_neg, correct_pos, correct_neg = val_stats
        if total_pos > 0:
            precision_pos = correct_pos / max(1, (correct_pos + (total_neg - correct_neg)))
            recall_pos = correct_pos / total_pos
            print(f"  Positive: {correct_pos}/{total_pos} (recall: {recall_pos:.4f})")
        if total_neg > 0:
            precision_neg = correct_neg / max(1, (correct_neg + (total_pos - correct_pos)))
            recall_neg = correct_neg / total_neg
            print(f"  Negative: {correct_neg}/{total_neg} (recall: {recall_neg:.4f})")
        print(f"  Label distribution: {total_pos} positive, {total_neg} negative")
        print()

        # Log data.
        train_loss.append(train_total_loss)  # 记录训练损失
        val_loss.append(val_total_loss)  # 记录验证损失
        train_n_correct_list.append(train_accuracy)  # 记录训练准确率
        val_n_correct_list.append(val_accuracy)  # 记录验证准确率

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
        writer.add_scalar('Accuracy/train', train_accuracy, n)  # 记录训练准确率到TensorBoard
        writer.add_scalar('Accuracy/test', val_accuracy, n)  # 记录验证准确率到TensorBoard
