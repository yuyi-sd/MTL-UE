"""from Senser and Koltun git repo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator
from itertools import chain
import math

"""
Adapted from: https://github.com/nik-dim/pamal/blob/master/src/models/factory/resnet.py
"""

class BasicBlock(nn.Module):
    """BasicBlock block for the Resnet. Adapted from official Pytorch source code."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetMTL(nn.Module):
    """
    ResNet architecture adapted from official Pytorch source code.
    The main difference lies in replacing the last FC layer dedicated for classification
    with a final FC layer that will be the shared representation for MTL
    """

    def __init__(self, n_tasks=3, num_blocks=(2, 2, 2, 2),
                 task_outputs=(9, 2, 5), in_channels=3, activation="elu"):
        super(ResNetMTL, self).__init__()

        self.n_tasks = n_tasks
        self.in_planes = 64
        self.in_channels = in_channels
        self.task_outputs = task_outputs
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_task_heads()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_task_heads(self):
        for i in range(len(self.task_outputs)):
            setattr(self, f"head_{i}", torch.nn.Linear(256, self.task_outputs[i]))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = [getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)]
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv1.parameters(),
            self.bn1.parameters(),
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()


class ResNet50MTL(nn.Module):
    """
    ResNet architecture adapted from official Pytorch source code.
    The main difference lies in replacing the last FC layer dedicated for classification
    with a final FC layer that will be the shared representation for MTL
    """

    def __init__(self, n_tasks=3, num_blocks=(3, 4, 6, 3),
                 task_outputs=(9, 2, 5), in_channels=3, activation="elu"):
        super(ResNet50MTL, self).__init__()

        self.n_tasks = n_tasks
        self.in_planes = 64
        self.in_channels = in_channels
        self.task_outputs = task_outputs
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_task_heads()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_task_heads(self):
        for i in range(len(self.task_outputs)):
            setattr(self, f"head_{i}", torch.nn.Linear(256, self.task_outputs[i]))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = [getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)]
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv1.parameters(),
            self.bn1.parameters(),
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()

class ResNetMTL_binary(nn.Module):
    """
    ResNet architecture adapted from official Pytorch source code.
    The main difference lies in replacing the last FC layer dedicated for classification
    with a final FC layer that will be the shared representation for MTL
    """

    def __init__(self, n_tasks=3, num_blocks=(2, 2, 2, 2), in_channels=3, activation="elu"):
        super(ResNetMTL_binary, self).__init__()

        self.n_tasks = n_tasks
        self.in_planes = 64
        self.in_channels = in_channels
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_task_heads()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_task_heads(self):
        for i in range(self.n_tasks):
            setattr(self, f"head_{i}", torch.nn.Linear(256, 1))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def regularization_loss(self):
        W = []
        for i in range(self.n_tasks):
            linear_params = getattr(self, f"head_{i}").weight
            W.append(linear_params/(linear_params.norm(p=2, dim=0)+1e-8))
        W = torch.cat(W, dim = 1)
        inner_products = torch.matmul(W.T, W)
        identity_matrix = torch.eye(W.shape[-1]).cuda()
        loss = F.mse_loss(inner_products, identity_matrix)
        return loss

    def forward(self, x, return_representation=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = torch.cat([getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1)
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv1.parameters(),
            self.bn1.parameters(),
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()


class ResNet50MTL_binary(nn.Module):
    """
    ResNet architecture adapted from official Pytorch source code.
    The main difference lies in replacing the last FC layer dedicated for classification
    with a final FC layer that will be the shared representation for MTL
    """

    def __init__(self, n_tasks=3, num_blocks=(3, 4, 6, 3), in_channels=3, activation="elu"):
        super(ResNet50MTL_binary, self).__init__()

        self.n_tasks = n_tasks
        self.in_planes = 64
        self.in_channels = in_channels
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_task_heads()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_task_heads(self):
        for i in range(self.n_tasks):
            setattr(self, f"head_{i}", torch.nn.Linear(256, 1))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def regularization_loss(self):
        W = []
        for i in range(self.n_tasks):
            linear_params = getattr(self, f"head_{i}").weight
            W.append(linear_params/(linear_params.norm(p=2, dim=0)+1e-8))
        W = torch.cat(W, dim = 1)
        inner_products = torch.matmul(W.T, W)
        identity_matrix = torch.eye(W.shape[-1]).cuda()
        loss = F.mse_loss(inner_products, identity_matrix)
        return loss

    def forward(self, x, return_representation=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = torch.cat([getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1)
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv1.parameters(),
            self.bn1.parameters(),
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()



class DenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet121MTL_binary(nn.Module):
    def __init__(self, n_tasks=3, in_channels=3, activation="elu", block=DenseBlock, nblocks=[6, 12, 24, 16], growth_rate=32, reduction=0.5):
        super(DenseNet121MTL_binary, self).__init__()
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.activation = activation
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(self.in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_planes, 256)
        self._init_task_heads()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def _init_task_heads(self):
        for i in range(self.n_tasks):
            setattr(self, f"head_{i}", torch.nn.Linear(256, 1))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.avgpool(F.relu(self.bn(out)))
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = torch.cat([getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1)
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv1.parameters(),
            self.dense1.parameters(),
            self.dense2.parameters(),
            self.dense3.parameters(),
            self.dense4.parameters(),
            self.trans1.parameters(),
            self.trans2.parameters(),
            self.trans3.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()



class DenseNet121MTL(nn.Module):
    def __init__(self, n_tasks=3, task_outputs=(9, 2, 5), in_channels=3, activation="elu", block=DenseBlock, nblocks=[6, 12, 24, 16], growth_rate=32, reduction=0.5):
        super(DenseNet121MTL, self).__init__()
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.activation = activation
        self.task_outputs = task_outputs
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(self.in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_planes, 256)
        self._init_task_heads()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def _init_task_heads(self):
        for i in range(len(self.task_outputs)):
            setattr(self, f"head_{i}", torch.nn.Linear(256, self.task_outputs[i]))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.avgpool(F.relu(self.bn(out)))
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = torch.cat([getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1)
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv1.parameters(),
            self.dense1.parameters(),
            self.dense2.parameters(),
            self.dense3.parameters(),
            self.dense4.parameters(),
            self.trans1.parameters(),
            self.trans2.parameters(),
            self.trans3.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()



class VGG16MTL_binary(nn.Module):
    def __init__(self, n_tasks=3, in_channels=3, activation="elu", cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], init_weights=True):
        super(VGG16MTL_binary, self).__init__()
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.activation = activation
        self.features = self.make_layers(cfg, batch_norm=True, in_channels = self.in_channels)
        self.linear = nn.Linear(512, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_task_heads()
        if init_weights:
            self._initialize_weights()

    def _init_task_heads(self):
        for i in range(self.n_tasks):
            setattr(self, f"head_{i}", torch.nn.Linear(256, 1))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = torch.cat([getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1)
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.features.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class VGG16MTL(nn.Module):
    def __init__(self, n_tasks=3, task_outputs=(9, 2, 5), in_channels=3, activation="elu", cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], init_weights=True):
        super(VGG16MTL_binary, self).__init__()
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.activation = activation
        self.task_outputs = task_outputs
        self.features = self.make_layers(cfg, batch_norm=True, in_channels = self.in_channels)
        self.linear = nn.Linear(512, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_task_heads()
        if init_weights:
            self._initialize_weights()

    def _init_task_heads(self):
        for i in range(len(self.task_outputs)):
            setattr(self, f"head_{i}", torch.nn.Linear(256, self.task_outputs[i]))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = torch.cat([getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1)
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.features.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTBMTL_binary(nn.Module):
    def __init__(self, n_tasks=3, activation="elu", image_size=128, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.n_tasks = n_tasks
        self.activation = activation
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.linear = nn.Linear(dim, 256)
        self._init_task_heads()

    def _init_task_heads(self):
        for i in range(self.n_tasks):
            setattr(self, f"head_{i}", torch.nn.Linear(256, 1))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        out = self.to_latent(x)
        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = torch.cat([getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1)
        if return_representation:
            return logits, features
        return logits


