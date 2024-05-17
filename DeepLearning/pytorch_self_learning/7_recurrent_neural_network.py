import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
'''
这里其实相当于一个序列的每一个节点就是一个行or列的处理
输入的是28维的向量,对应照片里面的行or列
然后序列一共28个LSTM单元代表有28个这样的行or列
'''
sequence_length = 28
input_size = 28
hidden_size = 128
# hiddem_size这个参数定义了a(这是吴恩达的notation, 这个程序是h) & c(cell)的维度
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

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

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 这里使用的单元是LSTM UNIT
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定义了一个全连接层, 输入的是cell的维度, 输出的是10种可能性
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        '''
        在循环神经网络中,特别是在长短期记忆网络中,h0(初始隐藏状态)和c0(初始细胞状态)是网络开始处理输入序列之前的状态。
        这些初始状态通常是零向量，意味着在时间序列的第一个时间步之前，网络没有先验知识。
        '''
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        '''
        上面这两个全是0的向量的定义中, num_layers代表RNN网络的深度
        x.size(0)其实代表的是batch_size, 因为记住, 模型中的一步步传播, 张量的第一个参数都是batch_size
        hidden_size是因为按照定义, 它的维度就是128维
        '''
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # 使用nn.LSTMd的时候， 输出的out的张量尺度是： 
        # tensor.Size([batch_size , sequence_length, hidden_size])
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # 这里:代表的是全部保留, 而-1代表的是取最后一个元素
        # 在这里的意义就是: 只关注序列最后一个单元所对应的深度的列的隐藏输出
        # 值得一提的是, 这里并没有使用softmax函数, 例如下面: 
        # out = nn.functional.softmax(self.fc(out[:, -1, :]), dim = 1)
        # 但是你还真别说, 用了softmax准确率没有不用高...
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 接下来的就是常规操作了 
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
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
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')