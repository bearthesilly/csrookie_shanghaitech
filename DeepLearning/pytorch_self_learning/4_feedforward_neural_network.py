import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 这行代码检查是否有可用的CUDA（GPU），如果有则使用GPU进行训练，否则使用CPU。
# Hyper-parameters 
input_size = 784 
'''
MNIST 数据集中的手写数字图像是 28x28 像素的。
由于这是一个全连接(fully connected)的神经网络，我们需要将每个图像展平成一个一维的向量，以便它可以作为网络的输入。
'''
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())
'''
download参数默认是False;如果是True,且没有在root里面找到数据集,那么就会自动下载 
transform=transforms.ToTensor()将数据转化为了张量
train: 一个布尔值，指示是否加载训练集。第一个调用中设置为 True 来加载训练数据，第二个调用中设置为 False 来加载测试数据。
'''
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
# 测试阶段，我们通常不需要打乱数据
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# 组合损失函数,将 nn.LogSoftmax 逻辑softmax层和负对数似然损失（negative log likelihood loss）组合在一起。
# 这个损失函数通常用于多分类问题。
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
# .parameters方法来自nn.Module; 包括权重（weights）和偏置（biases）
# 这些参数在模型训练过程中通过优化算法进行调整，以最小化损失函数。
# 大多数情况下，权重是随机生成的，而偏置默认为0或者也是随机值
# 且有requires_grad成员属性,代表着是否参与梯度计算
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        '''
        enumerate 返回一个由元组组成的迭代器，每个元组包含一对元素：索引和被 enumerate 作用的可迭代对象中对应的元素。
        每个元素实际上是一个批次的数据，通常是一个包含两个张量(tensor)的元组：(images, labels)
        '''
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        # torch.Size([100, 784]),为了能够放进nn.Linear
        labels = labels.to(device)
        # torch.Size([100])
        # Forward pass
        outputs = model(images)
        # torch.Size([100, 10])
        '''
        当你执行 model(images):
        PyTorch 查看 model 对象，发现它是一个 nn.Module 的子类实例。
        PyTorch 重载了 nn.Module 的 __call__ 方法，所以当你尝试以这种方式调用 model 时，它实际上调用了模型的 __call__ 方法。
        在 nn.Module 的 __call__ 实现中，当触发一个模型对象的调用时，它会自动寻找并执行模型类的 forward 方法。
        forward 方法是模型的前向传播逻辑所在，它接收输入数据 images,通过模型内部定义的层进行一系列计算,然后返回计算得到的输出 outputs。
        '''
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() # 防止梯度累加
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# 用于临时禁用在代码块内部的所有计算图和梯度计算。这通常用于模型的评估阶段
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        # torch.max(outputs, 1) 返回了两个张量：每个样本最大得分的张量和最大得分索引的张量
        total += labels.size(0) # 这里是100,batch_size
        correct += (predicted == labels).sum().item()
        '''
        发生了以下几步操作：
        predicted == labels: 这是一个比较操作,predicted 是模型预测的类别索引，而 labels 是数据集中的真实类别索引。
        这个操作会生成一个布尔类型的张量，其中的每个元素都是 True 或 False,取决于预测的类别索引是否与真实标签相匹配。
        .sum(): 这个函数会对布尔张量进行求和，其中 True 被当作 1,False 被当作 0。求和操作会计算出在当前批次中模型预测正确的样本数量。
        .item(): 这个函数将求和得到的标量张量转换为一个普通的Python数字(int 或 float)。
        由于 .sum() 返回的是一个张量，使用 .item() 可以将这个张量中的单个值提取出来,以便可以进行后续的Python算术操作。
        '''

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
'''
总结流程:
首先是利用torchvision.datasets.MNIST()(这里是MNIST)来生成train and test dataset
然后利用上面的datasets,通过torch.utils.data.DataLoader()方法生成train and test dataloader,注意batch_size and shuffle
之后创造模型类,继承nn.Module(注意super(NeuralNet, self).__init__());然后在自己的__init__定义fc,激活函数等
然后定义类的forward方法,即网络组成;同时注意从这里开始,.to(device)要纳入考量
定义完了类,创造一个model实例;紧接着配套上loss函数和优化器(注意model.parameters()来获得w b)
开始训练,一个epoch遍历完全部的数据一次;注意for i, (images, labels) in enumerate(train_loader)
然后获取images and labels,注意reshape和.to(device);model(images)训练一批次
一批次训练结束之后,优化器梯度清空,反向传播,优化器更新模型参数
最后评估,注意torch.max()方法和correct += (predicted == labels).sum().item()
以及非常重要的with torch.no_grad():,让梯度不再纳入计算的考量之中
'''