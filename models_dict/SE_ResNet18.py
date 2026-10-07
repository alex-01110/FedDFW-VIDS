import torch
import torch.nn.functional as F
from .reparam_function import ReparamModule
import numpy as np
import torch.nn as nn
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcite, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.se = SqueezeExcite(out_channels) if use_se else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEResNet18(ReparamModule):
    def __init__(self, num_classes=10):
        super(SEResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1, use_se=True)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=1, use_se=True)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=1, use_se=True)
        # self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2, use_se=True)
        # self.G1 = nn.Linear(768, 192)
        # self.G2 = nn.Linear(1536, 384)
        self.G3 = nn.Linear(3072, 768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * BasicBlock1D.expansion, num_classes)
        self.fc2 = nn.Linear(1344, num_classes)
    def _make_layer(self, block, out_channels, blocks, stride=1, use_se=False):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, use_se))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # # x = self.layer4(x)
        #
        # # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        #
        # return x
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        # o1 = output1.view(output1.size(0), -1)
        # o2 = output2.view(output2.size(0), -1)
        # o3 = output3.view(output3.size(0), -1)
        # g1 = self.G1(o1)
        # g2 = self.G2(o2)
        # g3 = self.G3(o3)
        x = self.avgpool(output3)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # fused = torch.cat((g1, g2, g3), dim=1)
        # x2 = self.fc2(fused)
        return x, output1, output2, output3
