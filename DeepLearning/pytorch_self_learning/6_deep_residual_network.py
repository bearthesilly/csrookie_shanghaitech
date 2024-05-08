# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
batch_size = 100
learning_rate = 0.001

# Image preprocessing modules
# 原来处理不复杂的时候,只有transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])
'''
transforms.Pad(4):这个操作会在图像的每个边缘(上、下、左、右)添加4个像素的填充。
填充的像素通常是复制边缘像素的值，这有助于在训练过程中增加图像的多样性，同时保持图像内容的完整性。
transforms.RandomHorizontalFlip()：这个操作以一定的概率随机地水平翻转图像。
水平翻转可以模拟镜像情况，从而增加数据集的多样性，有助于模型学习到更加鲁棒的特征。
transforms.RandomCrop(32):这个操作从原始图像中随机裁剪出一个32x32像素的区域。
裁剪操作可以模拟图像的不同视角，进一步增加数据集的多样性。

注意是transforms.Compose([])里面填充内容
'''

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=False)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
# 这里bias = False将不会加上偏置项,因为论文里面是这么说的
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # inplace=True是为了减少内存消耗
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        '''
        这段代码的意思是,如果downsample不为None,则先对输入x进行下采样操作,然后再与经过残差块其余部分的输出相加。
        downsample 不为 None 的情况通常包括：尺度减小的残差块,增加网络深度,不同尺度的残差块之间
        如果downsample为None,则直接将输入x作为残差连接的参考(因为尺度一样)
        '''
        out += residual
        out = self.relu(out)
        return out
'''
downsample=None通常指的是在残差块中不进行下采样(downsampling)操作
通常出现在残差块的定义中，表示该块不需要对输入进行空间尺寸的缩减。
这样，当残差块的输入和输出具有相同的空间维度时，恒等连接可以直接将输入添加到输出上，而不需要进行任何调整。

'''
# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0]) # 不改变尺度
        # layer[]代表残差块数量,对应[2,2,2]
        self.layer2 = self.make_layer(block, 32, layers[1], 2) # 改变尺度
        self.layer3 = self.make_layer(block, 64, layers[2], 2) # 改变尺度
        self.avg_pool = nn.AvgPool2d(8) # 8x8的平均池化层,因为32/2/2/8 = 1,可用于全连接层
        self.fc = nn.Linear(64, num_classes) # 全连接层
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            # 相当于是检查输入和输出的尺度是否一样
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # 第一个要实现尺度的转换,但是之后的残差块都是输入和输出尺度一样
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    # 使用 * 对 layers 列表进行解包，可以将列表中的每个模块作为独立的参数传递给 nn.Sequential 构造器
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        # 此时,out.size() 为 tensor.Size([100,1,1,64]),且1,1这两个维度都是同一个值(池化后的平均值)
        out = out.view(out.size(0), -1) # .view()方法和.reshape()很像
        out = self.fc(out)
        return out
    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
# 没有设置学习率调度器,那么优化器将使用创建时指定的固定学习率进行训练;因此要手动更新优化器的学习率
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate,防止过拟合
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
