import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16_bn


class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        # device指使用的设备，可以直接传入torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.feature_loss = loss

        # 下面5个assert是对传入的参数进行检查
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)
        self.weights = torch.tensor(weights).to(device)
        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        # 利用eval和requires_grad=False将权重冻结，方便输出特征图
        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        vgg = vgg.to(device)

        # bns为[4, 11, 21, 31, 41]，是5个BatchNorm2d层
        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        # 检查bns包含的是不是BN层
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        # 对于我们指定的blocks(需要取出哪几层的输出)，将相应bn层使用register_forward_hook方法来获取其输出
        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]

        # features其实就是一个精简的vgg16。我们需要哪几层的输出，就保留这几层之前的结构。如果我们只需要前两块的输出，那么后面三块其实就可以去掉
        # 了，减少运算量
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # 原代码
        # inputs = F.normalize(inputs, mean, std)
        # targets = F.normalize(targets, mean, std)

        # 上面原代码的改进代码
        transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        inputs = transformer(inputs)
        targets = transformer(targets)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        targets = targets.to(device)

        # extract feature maps
        # 得到输入图片经过3个选取的block的输出特征图
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        # 得到输入图片经过3个选取的block的输出特征图
        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0

        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs) * w

        return loss


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


def perceptual_loss(x, y):
    return F.mse_loss(x, y)


def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)