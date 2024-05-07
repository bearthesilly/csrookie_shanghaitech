import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            # 28*28*16, 16是卷积核的数量
            nn.BatchNorm2d(16), # 对每一个28*28进行一次normalization
            nn.ReLU(), # 激活函数,相当于是去掉了负数部分
            nn.MaxPool2d(kernel_size=2, stride=2)) # 每一个2*2卷积核里面挑最大的数
            #  最大池化层用于降低特征图的空间维度（高度和宽度），同时增加感受野（receptive field），有助于提取图像的高级特征
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out.size(0)就是batch_size
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # 不reshape的原因是:输入的格式就是[batch_size,channels(1),长(28),宽(28)]
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
'''
model.eval()影响的主要组件是：
批量归一化层(Batch Normalization):在训练模式下,批量归一化层会使用来自当前小批次(mini-batch)的均值和方差来归一化输入数据。
这样做有助于加快训练过程，因为每个小批次的数据分布可能会有所不同。
然而,在评估模式下,如果使用model.eval()，批量归一化层会使用训练阶段计算的移动平均均值和方差，而不是当前小批次的统计量。
这意味着在评估或测试时，无论输入数据来自哪个小批次，批量归一化层都将使用相同的均值和方差，这有助于保持模型行为的一致性。

丢弃(Dropout)：在训练过程中，丢弃层会随机地关闭一些神经元，以防止过拟合。但在评估模式下，丢弃层将不再关闭任何神经元
因为此时我们希望模型以最佳性能运行，而不是模拟训练时的随机性。
'''
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

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model_cv.ckpt')