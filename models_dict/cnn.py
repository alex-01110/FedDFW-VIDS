import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torchvision
from six import add_metaclass
from torch.nn import init
import copy
import math
from .reparam_function import ReparamModule
from typing import List


    

class CNNIDS2017(nn.Module):
    def __init__(self):
        super(CNNIDS2017, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),             # 1维卷积
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(3328,1024),
            nn.ReLU(),
            nn.Linear(1024, 7)

        )

    def forward(self, x):
        x= self.module(x)
        return x


class CNNIDS2017_fedlaw(ReparamModule):
    def __init__(self, norm_method, device, num_classes=11, feature_dim=1024, 
                 num_groups=1, affine_group_norm=False, before_activation=False, 
                 return_feature=False, bias=False):
        super(CNNIDS2017_fedlaw, self).__init__()
        
        # 1. 网络层定义（拆分独立模块，保持原IDS2017网络维度）
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=bias)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=bias)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # 全连接层输入维度保持原网络的4864（适配IDS2017数据集特征）
        self.fc1 = nn.Linear(in_features=4864, out_features=1024, bias=bias)
        self.fc2 = nn.Linear(in_features=1024, out_features=feature_dim, bias=bias)
        self.w = nn.Linear(feature_dim, num_classes, bias=bias)  # 最终分类层
        
        # 2. 归一化相关参数
        self.before_activation = before_activation
        self.return_feature = return_feature
        
        # 3. 归一化层定义（对应4层：conv1→0, conv2→1, fc1→2, fc2→3）
        # 组归一化（GroupNorm）
        self.gn0 = nn.GroupNorm(num_groups, num_channels=64, affine=affine_group_norm).to(device)
        self.gn1 = nn.GroupNorm(num_groups, num_channels=256, affine=affine_group_norm).to(device)
        self.gn2 = nn.GroupNorm(num_groups, num_channels=1024, affine=affine_group_norm).to(device)
        self.gn3 = nn.GroupNorm(num_groups, num_channels=feature_dim, affine=affine_group_norm).to(device)
        
        # 批归一化（1D卷积用BatchNorm1d）
        self.bn0 = nn.BatchNorm1d(num_features=64, affine=affine_group_norm).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=256, affine=affine_group_norm).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=1024, affine=affine_group_norm).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=feature_dim, affine=affine_group_norm).to(device)
        
        # 归一化层字典映射
        self.group_norms = {0: self.gn0, 1: self.gn1, 2: self.gn2, 3: self.gn3}
        
        # 4. 可学习缩放参数
        self.gamma = nn.Parameter(
            data=torch.tensor(feature_dim **(1 / 2)), 
            requires_grad=True
        )
        
        # 5. 解析归一化方法，生成每层的归一化函数链
        processed_norm = norm_method.replace(" ", "").strip('][').split(',')
        self.normalization(processed_norm)
        
    def forward(self, x):
        # 根据配置选择激活前/后应用归一化
        if self.before_activation:
            x = self.feature_norm_before_activation(x)
        else:
            x = self.feature_norm_after_activation(x)
        
        feature = x  # 保存中间特征
        x = self.w(x)  # 最终分类输出
        
        if self.return_feature:
            return x, feature  
        else:
            return x 
    
    # 特征归一化基础方法（适配1D数据）
    def zero_mean(self, x):
        """对输入特征去均值"""
        if len(x.shape) == 2:  # 全连接层输出 (batch, feature)
            mean = torch.mean(x, dim=1)
            return x - mean[:, None]
        elif len(x.shape) == 3:  # 1D卷积层输出 (batch, channel, length)
            mean = torch.mean(x, dim=(1, 2), keepdim=False)
            return x - mean[:, None, None]
        else:
            raise NotImplementedError(f"不支持的维度: {x.shape}")
    
    def l2_norm(self, x, variance_scale=False):
        """对输入特征做L2归一化"""
        if len(x.shape) == 2:  # 全连接层
            normed = F.normalize(x, dim=1)
            if variance_scale:
                return normed * torch.sqrt(torch.tensor(x.shape[1], device=x.device))
            return normed
        elif len(x.shape) == 3:  # 1D卷积层
            normed = F.normalize(x, dim=(1, 2))  # 在通道和长度维度归一化
            if variance_scale:
                scale = torch.sqrt(torch.tensor(x.shape[1] * x.shape[2], device=x.device))
                return normed * scale
            return normed
        else:
            raise NotImplementedError(f"不支持的维度: {x.shape}")
    
    def normalization(self, method: List[str]) -> List:
        """解析归一化方法，为每层生成归一化函数链"""
        # 初始化4层的函数链（对应conv1→0, conv2→1, fc1→2, fc2→3）
        function_per_layer = [lambda x: x for _ in range(len(method))]
        
        for layer_num, method_layer in enumerate(method):
            for letter in method_layer:
                letter = letter.lower()
                if letter == 's':  # 去均值
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.zero_mean(f(x))
                elif letter == 'n':  # L2归一化（无缩放）
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.l2_norm(f(x))
                elif letter == 'v':  # L2归一化（有方差缩放）
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.l2_norm(f(x), variance_scale=True)
                elif letter == 'c':  # 乘以常数（解析数字）
                    multiplier = int(''.join([i for i in method_layer if i.isdigit()]))
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num], m=multiplier: f(x) * m
                elif letter == 'g':  # 组归一化
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num], ln=layer_num: self.group_norms[ln](f(x))
                elif letter == 'x':  # 最后一层的可学习参数缩放
                    if layer_num == 3:
                        function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.gamma * f(x)
                    else:
                        raise NotImplementedError("'x'仅支持最后一层（索引3）")
                elif letter.isdigit():  # 忽略数字（已在'c'中处理）
                    continue
                else:
                    raise NotImplementedError(f"不支持的归一化方法: {letter}")
        
        self.function_per_layer = function_per_layer
        return function_per_layer
    
    def feature_norm_before_activation(self, x):
        """归一化在激活函数前应用"""
        x = self.conv1(x)
        x = self.function_per_layer[0](x)  # conv1层归一化
        x = self.pool1(F.relu(x))
        
        x = self.conv2(x)
        x = self.function_per_layer[1](x)  # conv2层归一化
        x = self.pool2(F.relu(x))
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.function_per_layer[2](x)  # fc1层归一化
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.function_per_layer[3](x)  # fc2层归一化
        x = F.relu(x)
        
        return x
    
    def feature_norm_after_activation(self, x):
        """归一化在激活函数后应用"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.function_per_layer[0](x)  # conv1层激活后归一化
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.function_per_layer[1](x)  # conv2层激活后归一化
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.function_per_layer[2](x)  # fc1层激活后归一化
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.function_per_layer[3](x)  # fc2层激活后归一化
        
        return x




    
class MLPIDS2017_fedlaw(ReparamModule):
    def __init__(self):
            super(MLPIDS2017_fedlaw, self).__init__()

            self.module = nn.Sequential(
                nn.Linear(78, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 7),
    
            )


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.module(x)

        return x


class MLPIDS2017(nn.Module):
    def __init__(self):
        super(MLPIDS2017, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(78, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 7),
  
        )


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.module(x)

        return x
    



class CNN_FedLaw(ReparamModule):
    def __init__(self, norm_method, device, 
                 flatten_dim,        
                 num_classes,              # 分类数量，例如 2、5、11……
                 num_groups=1,
                 affine_group_norm=False,
                 before_activation=False,
                 return_feature=False,
                 bias=False):

        super(CNN_FedLaw, self).__init__()

        # 统一卷积结构
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=bias)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=3, padding=1, bias=bias)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()



        self.fc1 = nn.Linear(flatten_dim, 1024, bias=bias)
        self.w = nn.Linear(1024, num_classes, bias=bias)
      
        # 2. 归一化相关参数
        self.before_activation = before_activation
        self.return_feature = return_feature
        
        # 3. 定义组归一化（GroupNorm）和批归一化（BatchNorm）层
        # 对应网络层：conv1→0, conv2→1, fc1→2, fc2→3 
        self.gn0 = nn.GroupNorm(num_groups, num_channels=64, affine=affine_group_norm).to(device)
        self.gn1 = nn.GroupNorm(num_groups, num_channels=256, affine=affine_group_norm).to(device)
        self.gn2 = nn.GroupNorm(num_groups, num_channels=1024, affine=affine_group_norm).to(device)

        

        
        # 存储归一化层的字典（键为层索引）
        self.group_norms = {0: self.gn0, 1: self.gn1, 2: self.gn2}

        
        # 4. 可学习参数 
        self.gamma = nn.Parameter(
            data=torch.tensor(1024 **(1 / 2)), 
            requires_grad=True
        )
        
        # 5. 解析归一化方法，生成每层的归一化函数链
        processed_norm = norm_method.replace(" ", "").strip('][').split(',')
        self.normalization(processed_norm)
        
    def forward(self, x):
        # 根据配置选择激活前/后应用归一化
        if self.before_activation:
            x = self.feature_norm_before_activation(x)
        else:
            x = self.feature_norm_after_activation(x)
        
        feature = x  # 保存特征用于返回
        x = self.w(x)  # 最终分类
        
        if self.return_feature:
            return x , feature  
        else:
            return x 
    
    # 适配1D数据
    def zero_mean(self, x):
        """对输入特征去均值（适配1D卷积和全连接层）"""
        if len(x.shape) == 2:  # 全连接层输出 (batch, feature)
            mean = torch.mean(x, dim=1)
            return x - mean[:, None]
        elif len(x.shape) == 3:  # 1D卷积层输出 (batch, channel, length)
            mean = torch.mean(x, dim=(1, 2), keepdim=False)
            return x - mean[:, None, None]
        else:
            raise NotImplementedError(f"不支持的维度: {x.shape}")
    
    def l2_norm(self, x, variance_scale=False):
        """对输入特征做L2归一化（适配1D数据）"""
        if len(x.shape) == 2:  # 全连接层
            normed = F.normalize(x, dim=1)
            if variance_scale:
                return normed * torch.sqrt(torch.tensor(x.shape[1], device=x.device))
            return normed
        elif len(x.shape) == 3:  # 1D卷积层
            normed = F.normalize(x, dim=(1, 2))  # 在通道和长度维度归一化
            if variance_scale:
                scale = torch.sqrt(torch.tensor(x.shape[1] * x.shape[2], device=x.device))
                return normed * scale
            return normed
        else:
            raise NotImplementedError(f"不支持的维度: {x.shape}")
    
    def normalization(self, method: List[str]) -> List:
        """解析归一化方法，为每层生成归一化函数链"""
        # 初始化函数链（共3层：conv1→0, conv2→1, fc1→2）
        function_per_layer = [lambda x: x for _ in range(len(method))]
        
        for layer_num, method_layer in enumerate(method):
            for letter in method_layer:
                letter = letter.lower()
                if letter == 's':  # 去均值
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.zero_mean(f(x))
                elif letter == 'n':  # L2归一化（无缩放）
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.l2_norm(f(x))
                elif letter == 'v':  # L2归一化（有方差缩放）
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.l2_norm(f(x), variance_scale=True)
                elif letter == 'c':  # 乘以常数（解析数字）
                    multiplier = int(''.join([i for i in method_layer if i.isdigit()]))
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num], m=multiplier: f(x) * m
                elif letter == 'g':  # 组归一化
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num], ln=layer_num: self.group_norms[ln](f(x))
                elif letter == 'x':  # 最后一层的可学习参数缩放
                    if layer_num == 2:
                        function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.gamma * f(x)
                    else:
                        raise NotImplementedError("'x'仅支持最后一层（索引3）")
                elif letter.isdigit():  # 忽略数字（已在'c'中处理）
                    continue
                else:
                    raise NotImplementedError(f"不支持的归一化方法: {letter}")
        
        self.function_per_layer = function_per_layer
        return function_per_layer
    
    def feature_norm_before_activation(self, x):
        """归一化在激活函数前应用"""
        x = self.conv1(x)
        x = self.function_per_layer[0](x)  # 对conv1输出应用归一化
        x = self.pool1(F.relu(x))
        
        x = self.conv2(x)
        x = self.function_per_layer[1](x)  # 对conv2输出应用归一化
        x = self.pool2(F.relu(x))
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.function_per_layer[2](x)  # 对fc1输出应用归一化
        x = F.relu(x)
        
        
        return x
    
    def feature_norm_after_activation(self, x):
        """归一化在激活函数后应用"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.function_per_layer[0](x)  # 激活后归一化
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.function_per_layer[1](x)  # 激活后归一化
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.function_per_layer[2](x)  # 激活后归一化
        
        
        return x
    
class CNN_FedLaw_Standard(nn.Module):
    def __init__(self, norm_method, device, flatten_dim, num_classes, 
                 num_groups=1, affine_group_norm=False, 
                 before_activation=False, return_feature=False, bias=False):
        super(CNN_FedLaw_Standard, self).__init__()
        
        # === 复制 CNN_FedLaw 的结构定义 ===
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=bias)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=3, padding=1, bias=bias)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_dim, 1024, bias=bias)
        self.w = nn.Linear(1024, num_classes, bias=bias)
        # 2. 归一化相关参数
        self.before_activation = before_activation
        self.return_feature = return_feature
        
        # 3. 定义组归一化（GroupNorm）和批归一化（BatchNorm）层
        # 对应网络层：conv1→0, conv2→1, fc1→2, fc2→3 
        self.gn0 = nn.GroupNorm(num_groups, num_channels=64, affine=affine_group_norm).to(device)
        self.gn1 = nn.GroupNorm(num_groups, num_channels=256, affine=affine_group_norm).to(device)
        self.gn2 = nn.GroupNorm(num_groups, num_channels=1024, affine=affine_group_norm).to(device)

        

        
        # 存储归一化层的字典（键为层索引）
        self.group_norms = {0: self.gn0, 1: self.gn1, 2: self.gn2}

        
        # 4. 可学习参数 
        self.gamma = nn.Parameter(
            data=torch.tensor(1024 **(1 / 2)), 
            requires_grad=True
        )
        
        # 5. 解析归一化方法，生成每层的归一化函数链
        processed_norm = norm_method.replace(" ", "").strip('][').split(',')
        self.normalization(processed_norm)
        
    def forward(self, x):
        # 根据配置选择激活前/后应用归一化
        if self.before_activation:
            x = self.feature_norm_before_activation(x)
        else:
            x = self.feature_norm_after_activation(x)
        
        feature = x  # 保存特征用于返回
        x = self.w(x)  # 最终分类
        
        if self.return_feature:
            return x , feature  
        else:
            return x 
    
    # 适配1D数据
    def zero_mean(self, x):
        """对输入特征去均值（适配1D卷积和全连接层）"""
        if len(x.shape) == 2:  # 全连接层输出 (batch, feature)
            mean = torch.mean(x, dim=1)
            return x - mean[:, None]
        elif len(x.shape) == 3:  # 1D卷积层输出 (batch, channel, length)
            mean = torch.mean(x, dim=(1, 2), keepdim=False)
            return x - mean[:, None, None]
        else:
            raise NotImplementedError(f"不支持的维度: {x.shape}")
    
    def l2_norm(self, x, variance_scale=False):
        """对输入特征做L2归一化（适配1D数据）"""
        if len(x.shape) == 2:  # 全连接层
            normed = F.normalize(x, dim=1)
            if variance_scale:
                return normed * torch.sqrt(torch.tensor(x.shape[1], device=x.device))
            return normed
        elif len(x.shape) == 3:  # 1D卷积层
            normed = F.normalize(x, dim=(1, 2))  # 在通道和长度维度归一化
            if variance_scale:
                scale = torch.sqrt(torch.tensor(x.shape[1] * x.shape[2], device=x.device))
                return normed * scale
            return normed
        else:
            raise NotImplementedError(f"不支持的维度: {x.shape}")
    
    def normalization(self, method: List[str]) -> List:
        """解析归一化方法，为每层生成归一化函数链"""
        # 初始化函数链（共3层：conv1→0, conv2→1, fc1→2）
        function_per_layer = [lambda x: x for _ in range(len(method))]
        
        for layer_num, method_layer in enumerate(method):
            for letter in method_layer:
                letter = letter.lower()
                if letter == 's':  # 去均值
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.zero_mean(f(x))
                elif letter == 'n':  # L2归一化（无缩放）
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.l2_norm(f(x))
                elif letter == 'v':  # L2归一化（有方差缩放）
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.l2_norm(f(x), variance_scale=True)
                elif letter == 'c':  # 乘以常数（解析数字）
                    multiplier = int(''.join([i for i in method_layer if i.isdigit()]))
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num], m=multiplier: f(x) * m
                elif letter == 'g':  # 组归一化
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num], ln=layer_num: self.group_norms[ln](f(x))
                elif letter == 'x':  # 最后一层的可学习参数缩放
                    if layer_num == 2:
                        function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num]: self.gamma * f(x)
                    else:
                        raise NotImplementedError("'x'仅支持最后一层（索引3）")
                elif letter.isdigit():  # 忽略数字（已在'c'中处理）
                    continue
                else:
                    raise NotImplementedError(f"不支持的归一化方法: {letter}")
        
        self.function_per_layer = function_per_layer
        return function_per_layer
    
    def feature_norm_before_activation(self, x):
        """归一化在激活函数前应用"""
        x = self.conv1(x)
        x = self.function_per_layer[0](x)  # 对conv1输出应用归一化
        x = self.pool1(F.relu(x))
        
        x = self.conv2(x)
        x = self.function_per_layer[1](x)  # 对conv2输出应用归一化
        x = self.pool2(F.relu(x))
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.function_per_layer[2](x)  # 对fc1输出应用归一化
        x = F.relu(x)
        
        
        return x
    
    def feature_norm_after_activation(self, x):
        """归一化在激活函数后应用"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.function_per_layer[0](x)  # 激活后归一化
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.function_per_layer[1](x)  # 激活后归一化
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.function_per_layer[2](x)  # 激活后归一化
        
        
        return x
class MLP_NSLKDD_fedlaw(ReparamModule):
    def __init__(self):
            super(MLP_NSLKDD_fedlaw, self).__init__()

            self.module = nn.Sequential(
                nn.Linear(123, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 5),
    
            )


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.module(x)

        return x


class MLP_NSLKDD(nn.Module):
    def __init__(self):
        super(MLP_NSLKDD, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(123, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
  
        )


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.module(x)

        return x


class MLP_UNSW_fedlaw(ReparamModule):
    def __init__(self):
            super(MLP_UNSW_fedlaw, self).__init__()

            self.module = nn.Sequential(
                nn.Linear(196, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 9),
    
            )


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.module(x)

        return x


class MLP_UNSW(nn.Module):
    def __init__(self):
        super(MLP_UNSW, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(196, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 9),
  
        )


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.module(x)

        return x
    
class CNN_UNSW(nn.Module):
    def __init__(self):
        super(CNN_UNSW, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),             # 1维卷积
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(12544,1024),
            nn.ReLU(),
            nn.Linear(1024, 9)

        )

    def forward(self, x):
        x= self.module(x)
        return x
    
class CNN_carhacking(nn.Module):
    def __init__(self):
        super(CNN_carhacking, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),             # 1维卷积
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024, 9)

        )

    def forward(self, x):
        x= self.module(x)
        return x


class CNN_VeRemi(nn.Module):
    def __init__(self):
        super(CNN_VeRemi, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),             # 1维卷积
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024, 6)

        )

    def forward(self, x):
        x= self.module(x)
        return x

class CNN_TON_IoT(nn.Module):
    def __init__(self):
        super(CNN_TON_IoT, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),             # 1维卷积
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(2304,1024),
            nn.ReLU(),
            nn.Linear(1024, 10)

        )

    def forward(self, x):
        x= self.module(x)
        return x
     
class CNN_IOV2024(nn.Module):
    def __init__(self):
        super(CNN_IOV2024, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),             # 1维卷积
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024, 6)

        )

    def forward(self, x):
        x= self.module(x)
        return x

class CNN_IOTdataset(nn.Module):
    def __init__(self):
        super(CNN_IOTdataset, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),             # 1维卷积 (N, 64, 31)
            nn.ReLU(),
            nn.MaxPool1d(2),                                        # 池化层 (N, 64, 15)
            nn.Conv1d(64, 256, kernel_size=3, padding=1),           # 1维卷积 (N, 256, 15)
            nn.ReLU(),
            nn.MaxPool1d(2),                                        # 池化层 (N, 256, 7)
            nn.Flatten(),                                           # (N, 256 * 7) = (N, 1792)
            nn.Linear(1792, 1024),                                  # 全连接层映射到 1024
            nn.ReLU(),
            nn.Linear(1024, 9)                                      # 9类输出
        )
    def forward(self, x):
        x = self.module(x)
        return x

class CNN_EDGE_IIoT(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_EDGE_IIoT, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(10),  # 自适应池化锁定序列维为 10，完全免疫异构 CSV 长度突升突降报错
            nn.Flatten(),
            nn.Linear(256 * 10, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.module(x)

class MLP_EDGE_IIoT(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP_EDGE_IIoT, self).__init__()
        # 相比普通的 Flatten，这里也用自适应池化保证输入特征数量为 60 条，防止输入维度越界
        self.adapter = nn.Sequential(
            nn.AdaptiveMaxPool1d(60),
            nn.Flatten(),
            nn.Linear(60, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x.shape is expected to be [batch, 1, seq_len], adaptive pool will handle it
        return self.adapter(x)

class CNN_NSLKDD(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_NSLKDD, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(10),
            nn.Flatten(),
            nn.Linear(256 * 10, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.module(x)

class MLP_NSLKDD(nn.Module):
    def __init__(self, num_classes=5):
        super(MLP_NSLKDD, self).__init__()
        self.adapter = nn.Sequential(
            nn.AdaptiveMaxPool1d(40),
            nn.Flatten(),
            nn.Linear(40, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.adapter(x)

class MLP_VeRemi(nn.Module):
    def __init__(self, num_classes=6):
        super(MLP_VeRemi, self).__init__()
        self.adapter = nn.Sequential(
            nn.AdaptiveMaxPool1d(40),
            nn.Flatten(),
            nn.Linear(40, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.adapter(x)
