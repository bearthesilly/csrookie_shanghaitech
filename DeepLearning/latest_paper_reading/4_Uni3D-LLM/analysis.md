如何实现断点续传

断点续传有几个核心:

- 在初始化模型和优化器之后, 要判断是否之前的模型; 如果有, 那么需要加载检查点
- 加载检查点加载的内容包括模型本身(连带着参数全部都加在了), 优化器, 以及非常重要的``start_epoch``参数, 记录从第几次epoch开始训练
- 训练模型的过程中, 要阶段性(甚至是训练一轮就存储一轮)保存checkpoint. 注意如果是有一定的频率进行保存, 例如5, 10轮保存一次, 那么可能会造成多训练几轮的结果, 但是影响应该不会特别大

接下来介绍常用的函数, 代码

- 首先是在训练循环中保存checkpoint了: 首先我们要定义save_checkpoint函数, 然后训练循环中调用

````python
def save_checkpoint(state, filename='checkpoint.pth.tar'): 
    # 保存到同目录的'checkpoint.pth.tar'
    torch.save(state, filename)
    
for epoch in range(start_epoch, num_epoch):
    # ...
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer':optimizer.state_dict(),
    })
````

调用的时候, {}里面的字典构成了state, 然后被储存了起来; 有的时候储存的东西不止这么些, 例如scheduler

- 其次是加载保存好的检查点, 这里需要自己定义``load_checkpoint``函数, 然后再开始训练之前调用:

````python
def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        epoch = 0
    return model, optimizer, epoch

# 初始化模型、优化器和损失函数
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 加载检查点
model, optimizer, start_epoch = load_checkpoint(model, optimizer)
````

注意这个函数里面自动带了判断是否有上一个检查点, 更加方便了

那么这就是大致的pipeline了, 实战演练





