# PointNet

之前处理3D数据, 都是是利用: voxel grid 或者是二维呈现. 

![image](img/1.png)

二维呈现, 是为了将convolutional network运用再图像上面

那么Point net网络, 输入的数据形式是点云. 关于点云数据, 论文特别提到: 

Point clouds are simple and unified structures that avoid the combinatorial irregularities and complexities of meshes......still has to respect the fact that a point cloud is just a set of points and therefore ***invariant to permutations.*** 

为了结局组合不敏感(因为点换了位置, 依然整体信息是不变的), 论文使用了一种经典的对称函数, 最大池化

这个模型最后实现的是: 在分类任务上, 输入点云, 输出标签; 在场景分割上, 给每个点打上标签(例如, 这个点是属于椅子? 还是桌子?)

# model

附: pipeline，中文意为管线，意义等同于流水线。baseline意思是**基线**，这个概念是作为算法提升的参照物而存在的，相当于一个**基础模型**，可以以此为基准来比较对模型的改进是否有效。

![image](img/2.png)

首先要了解点云数据的特性, 论文论述了三个特点:

- Unordered. 每一个点只有三个坐标, 但是并没有引索! 
- Interaction among points. 点与点并不是孤立的, 而是像图像一样, pixel and pixel之间有关系, 蕴含着信息
- Invariance under transformation. 这个模型在空间中进行线性变换, 例如旋转, 对称等, 应该蕴含的信息是不会改变的

论文中认为, 模型中三个组件非常的关键: 

- Symmetric function to aggregate information

这是为了Unordered Input. 论文中提到:  For example, + and ∗ operators are symmetric binary functions. 因为自变量交换位置, 结果不变. 论文中也提到过, 曾经尝试使用过训练一个小模型去给它们简洁地排列, 但是表现非常差, 但是依然表现好过没有排序. 

所以说论文想找到一种函数, 满足: 

![image](img/3.png)

最后, 选择了maxpooling作为g函数

- Local and Global Information Aggregation

在模型图中可以看到, 后面将最大池化提取出来的全局特征与最大池化之前的局部特征进行拼接. 

`` Then we extract new per point features based on the combined point features - this time the per point feature is aware of both the local and global information.(from the paper)``

- Joint Alignment Network

`The semantic labeling of a point cloud has to be invariant if the point cloud undergoes certain geometric transformations`

为了把一个空间变换过的点云"摆正来", 论文提到了:

`Jaderberg et al. [9] introduces the idea of spatial transformer to align 2D images through sampling and interpolation`

`We predict an affine transformation matrix by a mini-network (T-net in Fig) and directly apply this transformation to the coordinates of input points`

当然, 这个训练出来的空间变换的矩阵应该长什么样子? 论文中提到, 希望最后训练出来的矩阵应该非常接近正交矩阵. 因为正交矩阵用作空间变换的最大特征就是: 长度和角度不变. 

![image](img/4.png)

因此使用了这个作为损失函数. 这种损失函数的设置其实能带来很多启发. 

最终, 论文提到: `Intuitively, our network learns to summarize a shape by a sparse set of key points`

# 代码

