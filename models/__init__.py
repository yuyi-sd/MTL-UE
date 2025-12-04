import mlconfig
import torch
import torch.nn as nn
import torchvision

from . import DenseNet, ResNet, ToyModel, inception_resnet_v1, resnet_official, ResNetMTL

class MultiCrossEntropyLoss:
    def __init__(self, weight=None):
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
    
    def __call__(self, logits_list, labels_list):
        total_loss = 0
        num_groups = len(logits_list)
        if self.weight is None:
            self.weight = [1.0] * num_groups

        for logits, labels, weight in zip(logits_list, labels_list, self.weight):
            loss = self.criterion(logits, labels)
            total_loss += loss * weight
        
        weighted_average_loss = total_loss / sum(self.weight)
        return weighted_average_loss

mlconfig.register(MultiCrossEntropyLoss)

mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
mlconfig.register(torch.nn.CrossEntropyLoss)
mlconfig.register(torch.nn.BCEWithLogitsLoss)

# Models
mlconfig.register(ResNet.ResNet)
mlconfig.register(ResNet.ResNet18)
mlconfig.register(ResNet.ResNet18AE)
mlconfig.register(ResNet.ResNet18VAE)
mlconfig.register(ResNet.ResNet18DVAE)
mlconfig.register(ResNet.ResNet18DAE)
mlconfig.register(ResNet.ResNet18DVAE_pure)
mlconfig.register(ResNet.ResNet18CVAE)
mlconfig.register(ResNet.RResNet18VAE)
mlconfig.register(ResNet.ResNet18NAE)
mlconfig.register(ResNet.ResNet18DVAElf)
mlconfig.register(ResNet.VUNet)
mlconfig.register(ResNet.VAE)
mlconfig.register(ResNet.ResNet34)
mlconfig.register(ResNet.ResNet50)
mlconfig.register(ResNet.ResNet101)
mlconfig.register(ResNet.ResNet152)
mlconfig.register(ToyModel.ToyModel)
mlconfig.register(DenseNet.DenseNet121)
mlconfig.register(inception_resnet_v1.InceptionResnetV1)
mlconfig.register(torchvision.models.vgg19)
mlconfig.register(torchvision.models.mobilenet_v2)
mlconfig.register(torchvision.models.vit_b_16)
# torchvision models
# mlconfig.register(torchvision.models.resnet18)
mlconfig.register(resnet_official.resnet18)
# mlconfig.register(torchvision.models.resnet50)
# mlconfig.register(torchvision.models.densenet121)
mlconfig.register(ResNetMTL.ResNetMTL)
mlconfig.register(ResNetMTL.ResNetMTL_binary)
mlconfig.register(ResNetMTL.ResNet50MTL)
mlconfig.register(ResNetMTL.ResNet50MTL_binary)
mlconfig.register(ResNetMTL.DenseNet121MTL)
mlconfig.register(ResNetMTL.DenseNet121MTL_binary)
mlconfig.register(ResNetMTL.VGG16MTL)
mlconfig.register(ResNetMTL.VGG16MTL_binary)
mlconfig.register(ResNetMTL.ViTBMTL_binary)

# CUDA Options
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


@mlconfig.register
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


@mlconfig.register
class CutMixCrossEntropyLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)
