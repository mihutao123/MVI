import torch.nn as nn
import torch 
import math
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)        #全局平均池化，输入BCHW -> 输出 B*C*1*1
        self.Linear1 = nn.Linear(channel, channel // reduction, bias=False)
        self.mish = Mish()
        self.Linear2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        #self.fc = nn.Sequential(
            #nn.Linear(channel, channel // reduction, bias=False),   #可以看到channel得被reduction整除，否则可能出问题
            #nn.ReLU(inplace=True),
            #nn.Linear(channel // reduction, channel, bias=False),
            #nn.Sigmoid()
        #)
 
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)     #得到B*C*1*1,然后转成B*C，才能送入到FC层中。
        y = self.Linear1(y)
        y = self.mish(y)
        y = self.Linear2(y)
        y = self.sigmoid(y).view(b, c, 1)
        #y = self.fc(y).view(b, c, 1)     #得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        return x * y.expand_as(x)           #先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘。


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction =16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.se = SELayer(planes*4, reduction)
        #self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)
        #out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.mish(out)
        #out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.mish = Mish()
        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)
        #out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.mish(out)
        #out = self.relu(out)

        return out


class SEdetnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A', reduction =16):
        super(SEdetnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)
        self.se = SELayer(self.expansion*planes, reduction)
        self.mish = Mish()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.bn2(self.conv2(out)))
        out = self.mish(self.bn1(self.conv1(x)))
        out = self.mish(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.downsample(x)
        out = self.mish(out)
        #out = F.relu(out)
        return out



class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)
        self.mish = Mish()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out



class Neck(nn.Module):
    def __init__(self, planes = 128):
        super(Neck, self).__init__()
        self.conv1 = nn.Conv1d(planes*2, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)        
        self.conv3 = nn.Conv1d(planes, planes*4, kernel_size=1,stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*4) 
        self.se1 = SELayer(planes*4, 16)
        
        self.maxpool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv4 = nn.Conv1d(planes*8, planes, kernel_size=1,stride=1, bias=False)
        self.bn4 = nn.BatchNorm1d(planes)
        self.conv5 = nn.Conv1d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(planes)        
        self.conv6 = nn.Conv1d(planes, planes*8, kernel_size=1,stride=1, bias=False)
        self.bn6 = nn.BatchNorm1d(planes*8) 
        self.se2 = SELayer(planes*8, 16)         
        
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.mish = Mish()

    def forward(self, input1, input2, input3):
        
        x1 = self.mish(self.bn1(self.conv1(input1)))
        x2 = self.mish(self.bn2(self.conv2(x1)))
        x3 = self.mish(self.se1(self.bn3(self.conv3(x2))))
        x4 = self.maxpool1(x3)
        
        x5 = torch.cat([x4, input2], dim=1)
        
        x6 = self.mish(self.bn4(self.conv4(x5)))
        x7 = self.mish(self.bn5(self.conv5(x6)))
        x8 = self.mish(self.se2(self.bn6(self.conv6(x7))))
        x9 = self.maxpool2(x8)
        x10 = torch.cat([x9, input3], dim=1)

        return x10



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=6):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(14, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64,layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer4_1 = Neck(64)
        # self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_detnet_layer(in_channels=512)
        # self.avgpool = nn.AvgPool2d(14) #fit 448 input size
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_end = nn.Conv1d(64, 24, kernel_size=3, stride=2, padding=1, bias=False)
        #self.bn_end = nn.BatchNorm1d(12)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_detnet_layer(self,in_channels):
        layers = []
        layers.append(SEdetnet_bottleneck(in_planes=in_channels, planes=64, block_type='B'))
        layers.append(SEdetnet_bottleneck(in_planes=64, planes=64, block_type='A'))
        layers.append(SEdetnet_bottleneck(in_planes=64, planes=64, block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.mish(x1)
        x1 = self.maxpool(x1)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        #x6 = self.layer4_1(x3, x4, x5)
        x7 = self.layer5(x5)        
        #x6 = self.layer4_1(x3, x4, x5)
        #x7 = self.layer5(x6)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x8 = self.conv_end(x7)
        #x8 = self.bn_end(x8)
        #x8 = F.sigmoid(x8)  
        # x = x.view(-1,7,7,30)
        x8 = x8.permute(0,2,1)  

        return x8


def resnet_new(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model
