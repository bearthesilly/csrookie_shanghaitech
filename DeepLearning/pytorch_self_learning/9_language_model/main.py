# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 128 # 潜空间维度 
hidden_size = 1024 # 隐藏状态维度 
num_layers = 1 # 只有一层的LSTM单元构成的RNN
num_epochs = 5
num_samples = 1000     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('C:/Users/23714/Desktop/Shanghai Tech/4VDLab/pytorch自学/pytorch-tutorial-master/tutorials/02-intermediate/language_model/data/train.txt', batch_size)
# 注意这里的路径应该是绝对路径
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length


# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 投射到前潜空间里面
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # LSTM函数传入的两个参数, 第一个是输入的文本, 第二个是(h(隐藏状态), c(记忆细胞))
        self.linear = nn.Linear(hidden_size, vocab_size)
    '''
    nn.LSTM:这是PyTorch提供的一个LSTM层的类。
    embed_size:这是输入特征的维度,通常对应于输入向量的大小,比如经过嵌入层(embedding layer)处理后的词向量维度。
    hidden_size:这是隐藏层的维度,即LSTM内部处理信息的神经元数量。这个值决定了模型的容量,较大的隐藏层尺寸可以捕获更复杂的模式
    但也可能需要更多的数据和计算资源。
    num_layers:这是LSTM堆叠的层数。多层LSTM可以捕获更深层次的依赖关系,但同样,过多的层数可能导致计算成本增加和过拟合。
    batch_first=True:这个参数指定了输入和输出张量的第一个维度是批次大小(batch size)。
    当设置为 True 时,输入和输出张量的形状为 (batch, sequence, feature)，其中 batch 是批次大小,sequence 是序列长度,feature 是特征维度（在这里是 hidden_size)。
    如果设置为 False,则张量的形状为 (sequence, batch, feature)。
    '''
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        # 输出部分包含两个部分, 第一个是输出序列, 第二个是最终的隐藏状态和单元状态 
        # batch_first=True，形状为 (batch, sequence, hidden_size)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time steps
        # 线性层的输出可以被视为词汇表上每个单词的未经 softmax 处理的 logits(或称为得分)
        out = self.linear(out)
        return out, (h, c)
'''
可能会好奇, 这里RNNMM模型中完全没有提到分析的sequence的长度, 那么模型是如何知道的呢? 
其实关键就在, sequence length的信息是通过x张量长度知道的
'''
model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 
# 相当于是复制出来一个样本, 然后对这个样本进行任何操作都不会影响原来的张量
# Train the model
for epoch in range(num_epochs):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        # seq_length（序列长度）这个参数通常代表处理的序列中的元素个数
        inputs = ids[:, i:i+seq_length].to(device)
        # 通过i:i+seq_length的信息, 揣测下面的seq_length的内容
        # 因此, 直接用(i+1):(i+1)+seq_length的内容作为ground truth
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)
        
        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        '''
        clip_grad_norm_ 是 PyTorch 中的一个函数，用于裁剪（限制）模型参数梯度的大小。
        这是一种常用的正则化技术，用于防止梯度爆炸问题，特别是在训练深度神经网络时。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 的具体作用如下：
        参数: model.parameters() 返回模型 model 中所有的参数。
        裁剪值: 第二个参数 0.5 表示梯度的最大范数。在这个例子中,任何参数梯度的范数如果超过了0.5
        将会被缩放,使得其范数等于0.5。
        操作: 这个函数直接修改传入的参数梯度，而不是返回一个新的梯度副本。
        范数: 默认情况下，此函数使用的是 L2 范数(欧几里得范数)，也可以通过 norm_type 参数指定其他类型的范数。
        梯度爆炸问题: 在某些类型的网络(特别是循环神经网络 RNN 和长短期记忆网络中),梯度可能会随着时间步的增加而指数级增长
        导致训练不稳定。裁剪梯度可以防止这种情况。
        正则化: 梯度裁剪也可以视为一种正则化形式，它有助于促进模型的泛化能力。
        '''
        optimizer.step()

        step = (i+1) // seq_length
        # 每训练100个sequence进行一次汇报 
        if step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

# Test the model
with torch.no_grad():
    with open('"C:/Users/23714/Desktop/sample.txt"', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
        '''
        创建概率分布:prob = torch.ones(vocab_size) 创建了一个长度为 vocab_size 的张量，其中每个元素都初始化为 1。
        这意味着每个索引（代表词汇表中的一个单词）都被赋予了相同的初始概率。
        抽样:torch.multinomial(prob, num_samples=1) 使用 prob 作为概率分布，从中抽取一个样本。
        因为 prob 中所有元素都是相同的，所以这个操作实质上是从词汇表中随机均匀地选择一个单词索引。
        增加维度:.unsqueeze(1) 在抽样结果上增加了一个维度。如果抽样结果是一个一维张量（只有单词索引），增加维度后它将变成一个二维张量
        这通常是为了满足模型输入的维度要求。
        为什么一个单词可以? 每个单词经过LSTM单元处理后,不仅会产生一个关于下一个单词的预测结果
        而且这个结果(或其嵌入表示)还会直接用作下一个时间步LSTM单元的输入
        '''
        for i in range(num_samples):
            # Forward propagate RNN 
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp() # softmax
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)
            # input.fill_(word_id) 的操作可以被看作是在更新条件概率中的“条件”
            # input 的尺寸始终是 [1, 1], 相当于是输入换成了上一个预测生成的单词 
            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))

# Save the model checkpoints
torch.save(model.state_dict(), 'model.ckpt')