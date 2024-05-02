# pytorch

## 基本

````python
import torch
torch.version.__version__
# 2.1.0+cu121
a = torch.ones(3,3)
a
# tensor([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.]])
b = torch.ones(3,3)
a+b
#  tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
torch.cuda.is_available()
# True
a = a.to('cuda')
b = b.to('cpu')
a+b # 报错
````

## Chapter2

````python
from torchvision import models
from torchvision import transforms # 变形, 图片大小不一, 色彩编码方式不一
dir(models) # 查看所有支持的模型
alexnet=models.AlexNet()
resnet=models.resnet101(pretrained=True) # 加载两个模型
from torchvision import transforms
preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.Resize(224), #  resnet101标准尺寸
    transforms.ToTensor(), # array之类的要转化为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 背, RGB
        std=[0.229, 0.224, 0.225]
    )
])
from PIL import Image
img=Image.open("/content/drive/MyDrive/Colab Notebooks/dlwpt-code-master/data/p1ch2/bobby.jpg") # 因为我使用的是colab的谷歌网盘
img # 展示图片
img_t=preprocess(img) # 预处理
# torch.Size([3, 224, 398])
img_t.shape # 获取size
import torch
batch_t=torch.unsqueeze(img_t,0)
# 这么做是因为在预训练模型中, 一般来说输入的是四维目标, 是四维batch
# 0就代表在第0引索数字后面加上1
batch_t.shape
# torch.Size([1, 3, 224, 398]) 
resnet.eval() # 把网络从"不知道什么模式"变成"推理模式"
output=resnet(batch_t)
output.shape  # torch.Size([1, 1000])  变成了1000维的张量
# 其中1对应的是batchsize, 1000对应的是1000Imagenet种类
# 那么1000个类是什么东西呢
with open('/content/drive/MyDrive/Colab Notebooks/dlwpt-code-master/data/p1ch2/imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]
# 可以查看列表了, 这里就不看了
# 那么对应的最有可能的是什么呢? 
print(labels[output.argmax()])
# golden retriever
# 或者用另一种方式找到最大引索
_, index=torch.max(output, 1) # 1代表找最大的, 第一个输出的参数是最大值, 但是不需要
# 注意, 这里的index并不是一个数字, 而是tensor([207])
# 接下来介绍一些有意思的
output.sum()  # tensor(0.0035, grad_fn=<SumBackward0>)  说明并不是分布概率!
percentage=torch.nn.functional.softmax(output, dim=1)[0]*100 # *100, 1变成100
percentage.sum() # tensor(100.0000, grad_fn=<SumBackward0>)  终于转化为了正常的概率分布
# 显示前五个最大的可能性
_, indices = torch.sort(output, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
````

## Chapter3

````Python
import torch
a=torch.ones(3)
# tensor([1., 1., 1.]) 这个是1维的
# a = torch.ones(3, 4)
# tensor([[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]])  这个是2维的张量
a[1]  #tensor(1.)还是张量
float(a[1]) # 1.0  真正的数字
a[2] = 2.0  # 改变张量中的一个数据, 这和列表的操作有点像
points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0]) # 手动创建一个张量
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.shape  # points.shape
points[0,0] # tensor(4.)
# 内存相关
points.storage()
'''
4.0
 1.0
 5.0
 3.0
 2.0
 1.0
[torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 6]'''
second_point = points[1] # 非常特殊, 其实这个second_point一直指向的是points, 没有另开内存
# 而且如果对它进行了修改, points会对应被修改!!
# 启示: 如果要复制, 不能直接拷贝! 正确方法见下
another_point = points[1].clone()
second_point.storage_offset() # 输出是2, 因为第二行的首地址是引索2
another_point.storage_offset() # 输出是0, 说明和points在内存上没有任何关系
# 获取大小, points.size()和points.shape等价; 注意返回的是张量属性

# 介绍stride
points.stride()  # (2, 1)
# stride第一个: 从一行第一个调到下一行数字在storage中走的步数
# stride第二个指的是从一行第一个走到第二个数字在storage中走的步数
# 其实指的就是维度夸一格走的步数
points_t = points.t()  # 转置, 注意这里不是克隆, 仅仅是拷贝!
id(points.storage()) == id(points_t.storage()) # True, 作证了上面这一点
points_t.stride()  # (1, 2)

some_t = torch.ones(3, 4, 5)
some_t.stride() # (20, 5, 1) 第一维度走一个, 下面全部都要走完, 因此是20
# 第二个维度走一个, 下面一个走完, 因此是5; 最后自己的走一个就是1
# 是否连续: 主要看的就是遍历元素在内存中是不是按顺序的, 如果是反复横跳的话就是False
points.is_contiguous() # True
points_t.is_contiguous() # False
````

![image](pytorch/1.png)

注意这一部分其实挺关键的, 假如说转化为int的话, 直接32位, 内存拉满; 能用int8就int8

````Python
import torch
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
short_points.dtype
# torch.int16
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)
points_64 = torch.rand(5, dtype=torch.double)
# tensor([0.9812, 0.0012, 0.9305, 0.2557, 0.7876], dtype=torch.float64)
points_short = points_64.to(torch.short)
# to转化类型函数十分重要!
# tensor([0., 0., 0., 0., 0.], dtype=torch.float64) 转化为整数, 只会取整数部分
````

那么张量的引用, 和numpy或者list非常像

````python
import torch
some_list = list(range(6))
print(some_list[:])    # <1>
print(some_list[1:4])   # <2>
print(some_list[1:])    # <3>
print(some_list[:4])    # <4>
print(some_list[:-1])   # <5>
print(some_list[1:4:2]) # <6>
'''
[0, 1, 2, 3, 4, 5]
[1, 2, 3]
[1, 2, 3, 4, 5]
[0, 1, 2, 3]
[0, 1, 2, 3, 4]
[1, 3]
'''
print(points[1:] )      # <1>
print(points[1:, :])  # <2>
print(points[1:, 0])    # <3>
print(points[None]) # 相当于unsqueeze
print(points[None].size())
'''
tensor([[5., 3.],
        [2., 1.]])
tensor([[5., 3.],
        [2., 1.]])
tensor([5., 2.])
tensor([[[4., 1.],
         [5., 3.],
         [2., 1.]]])
torch.Size([1, 3, 2])
'''

points = torch.ones(3, 4)
points_np = points.numpy() # 张量转成numpy
'''
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)
'''
points = torch.from_numpy(points_np)  # numpy转成张量
'''
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
'''
````

文件读写

````python
import torch
# write in 
points = torch.zeros(3, 4)
torch.save(points, 'xxxx.t')
# 或者:
with open('xxx.t', 'wb') as f:
    torch.save(points, f)
# 模式代码见下图

a=torch.load('xxx.t')
with open('xxx.t', 'rb') as f:
    a = torch.load(f)
    
import h5py
f = h5py.File('xxx.hdf5', 'w')
dset=f.create_dataset('coords', data=points.numpy())
f.close()
f = h5py.File('xxx.hdf5', 'r')
dset = f['coords']
dset
# <HDF5 dataset "coords": shape (3, 4), type "<f4">
dset[:]
'''
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)
'''
f.close()
````

![image](pytorch/2.png)

````python
import torch
_ = torch.tensor([0.2126, 0.7152, 0.0722], names=['c'])
img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
weights = torch.tensor([0.2126, 0.7152, 0.0722])

batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns], batch指的多少图
img_gray_naive = img_t.mean(-3) # 以倒数第三个维度进行平均, 并且降维
batch_gray_naive = batch_t.mean(-3)
img_gray_naive.shape, batch_gray_naive.shape
# (torch.Size([5, 5]), torch.Size([2, 5, 5]))
weights = torch.tensor([0.2126, 0.7152, 0.0722])
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1) # 最后面加两位, 注意后面的下划线
# 后面加了两个维度, 那么就变成了5维, 这样就可以和平均后的图片进行矩阵乘法
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)
img_gray_weighted = img_weights.sum(-3) # 加权后相加
batch_gray_weighted = batch_weights.sum(-3)
img_weights.shape, batch_weights.shape, batch_t.shape, unsqueezed_weights.shape
'''
(torch.Size([3, 5, 5]),
 torch.Size([2, 3, 5, 5]),
 torch.Size([2, 3, 5, 5]),
 torch.Size([3, 1, 1]))
'''
# 爱因斯坦求和(容易出错, 后面很少用)
img_gray_weighted_fancy = torch.einsum('...chw,c->...hw', img_t, weights)
# chw, channel  height  weight; 在channel维度进行平均, 和height  weight无关
batch_gray_weighted_fancy = torch.einsum('...chw,c->...hw', batch_t, weights)
batch_gray_weighted_fancy.shape
# torch.Size([2, 5, 5])

weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
weights_named
# tensor([0.2126, 0.7152, 0.0722], names=('channels',))
img_named =  img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)
'''
img named: torch.Size([3, 5, 5]) ('channels', 'rows', 'columns')
batch named: torch.Size([2, 3, 5, 5]) (None, 'channels', 'rows', 'columns')
'''
weights_aligned = weights_named.align_as(img_named)
# 这是让一维的weights_named朝着三维的img_named对齐
weights_aligned.shape, weights_aligned.names
# (torch.Size([3, 1, 1]), ('channels', 'rows', 'columns'))

gray_plain = gray_named.rename(None)
gray_plain.shape, gray_plain.names
# (torch.Size([5, 5]), (None, None)) 重新命名
````

小练习:

![image](pytorch/3.png)

````python
import torch
a = torch.tensor(list(range(9)))
print(a.size(), a.storage_offset(), a.stride())
# torch.Size([9]) 0 (1,)   当然, 注意使torch.Size
b = a.view(3, 3)
id(b.storage()) == id(a.storage())  # False, 所以说是开了一个不同的内存空间
c = b[1:, 1:]
'''
tensor([[4, 5],
        [7, 8]])
'''
c.size(), c.storage_offset(), c.stride(), c.is_contiguous
# (torch.Size([2, 2]), 4, (3, 1), False)
````

![image](pytorch/4.png)

````python
import torch
a=torch.Tensor(list(range(9)))
# torch.cos(a)  这会报错! 显示是数据类型不对  当然现在好想这样操作是允许的
a = a.to(float)
a.dtype   # torch.float64
torch.cos(a)
'''
tensor([ 1.0000,  0.5403, -0.4161, -0.9900, -0.6536,  0.2837,  0.9602,  0.7539,
        -0.1455], dtype=torch.float64)
'''
torch.sqrt(a)
'''
tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000, 2.2361, 2.4495, 2.6458, 2.8284],
       dtype=torch.float64)
'''
````

## Chapter 4

读取三维ct扫描

````python
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)
# 第三句话是设置打印格式
import imageio
# DICOM是医学上非常常见的3D影像格式
dir_path = r"C:\Users\23714\Desktop\SKD时期文件\4VDLab\dlwpt-code-master\data\p1ch4\volumetric-dicom\2-LUNG 3.0  B70f-04083"
# 因为是三维的， 所以是volread, 命名也是用vol_arr
vol_arr = imageio.volread(dir_path, 'DICOM')
vol_arr.shape
# (99, 512, 512) 99是代表一共99张图片

vol = torch.from_numpy(vol_arr).float()
# 这一步：把arr转换成tensor，并且数据类型转化为float
vol = torch.unsqueeze(vol, 0)
# size最前面要加上1，为了模型训练
%matplotlib inline
import matplotlib.pyplot as plt
# 引入上面matplotlib是为了准备打印
plt.imshow(vol_arr[50])
# vol_arr应该有三个维度，上面代表的就是第50张横切面，见下面的第一张图
plt.imshow(vol_arr[:,256,:])
# 这张图片就是正面看过去的纵切面了！
````

![image](pytorch/5.png)

![image](pytorch/6.png)

接下来学习表格形式的读取

````python
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)
# 导入+设置打印格式
import csv
wine_path = r"C:\Users\23714\Desktop\SKD时期文件\4VDLab\dlwpt-code-master\data\p1ch4\tabular-wine\winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
                         skiprows=1)
# 注意到使用分号分隔的，但是其实CSV传统上是comma-separated的
# 注意到这里跳过了第一行，没错，第一行的代号是1不是0；跳过第一行之后下面的数据格式都能读取了
'''
array([[ 7.  ,  0.27,  0.36, ...,  0.45,  8.8 ,  6.  ],
       [ 6.3 ,  0.3 ,  0.34, ...,  0.49,  9.5 ,  6.  ],
       [ 8.1 ,  0.28,  0.4 , ...,  0.44, 10.1 ,  6.  ],
       ...,
       [ 6.5 ,  0.24,  0.19, ...,  0.46,  9.4 ,  6.  ],
       [ 5.5 ,  0.29,  0.3 , ...,  0.38, 12.8 ,  7.  ],
       [ 6.  ,  0.21,  0.38, ...,  0.32, 11.8 ,  6.  ]], dtype=float32)
'''
# 当然读取CSV还有其他方式
'''
import pandas as pd 
df = pd.read_csv(r"...path")
df.head()
'''
col_list = next(csv.reader(open(wine_path), delimiter=';'))
wineq_numpy.shape, col_list
# 这句话读取了第一行的特征属性
'''
((4898, 12),
 ['fixed acidity',
  'volatile acidity',
  'citric acid',
  'residual sugar',
  'chlorides',
  'free sulfur dioxide',
  'total sulfur dioxide',
  'density',
  'pH',
  'sulphates',
  'alcohol',
  'quality'])

'''
wineq = torch.from_numpy(wineq_numpy) # numpy array转换为tensor
wineq.shape, wineq.dtype
# (torch.Size([4898, 12]), torch.float32)
# 那么如果要深度学习，那么用这个如何制作训练集？输入的是前面的11个特征，输出的最后是quality
data = wineq[:, :-1] # <1>
data, data.shape
'''
(tensor([[ 7.00,  0.27,  ...,  0.45,  8.80],
         [ 6.30,  0.30,  ...,  0.49,  9.50],
         ...,
         [ 5.50,  0.29,  ...,  0.38, 12.80],
         [ 6.00,  0.21,  ...,  0.32, 11.80]]), torch.Size([4898, 11]))'''
target = wineq[:, -1] # <2>
target, target.shape
# (tensor([6., 6.,  ..., 7., 6.]), torch.Size([4898]))
# 还想要根据quality分数将酒分成很多类】
target.min(), target.max() # (tensor(3.), tensor(9.))
target_onehot = torch.zeros(target.shape[0], 10) # target.shape[0]就是为了拿到列元素数量
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
# target.unsqueeze(1)使得target变成torch.Size([4898, 1])
target_onehot[0:3], target[:3]
'''
(tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]),
 tensor([6, 6, 6]))
说明成功了，quality如果是6，那么第六列就是1，其他都是0，代表了这个酒的品质就是6
为什么要这么做？因为最后一层是一个softmax，输出的yhat的数据应该代表可能品质是这个的概率
'''
# 接下来尝试实现归一化，batch norm!(Deep Learning Concept)
data_mean = torch.mean(data, dim=0)
data_mean # torch.Size([11])
data_var = torch.var(data, dim=0)
data_var # torch.Size([11])
# 当然嫌麻烦的话，直接求方差，毕竟到时候除去的是方差
data_std = torch.std(data, dim=0)
data_std # torch.Size([11])
data_normalized = (data - data_mean) / torch.sqrt(data_var)
data_normalized

# 直观演示，假如说quality小于3的都认为是不好的酒
bad_indexes = target <= 3 # <1>
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum() # False=0, True=1
# (torch.Size([4898]), torch.bool, tensor(20))  只有20种品质不好

bad_data = data[bad_indexes]
bad_data.shape # 把不好的酒的11种参数都搞过来

bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)] # <1>
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
'''
这段代码使用了enumerate函数来遍历一个由zip函数组合的四个列表，其中col_list是列名列表，bad_mean、mid_mean和good_mean分别是包含坏、中等和好的意义的值的列表。在每次迭代中，enumerate函数会返回一个索引i和一个包含当前迭代中来自四个列表的元素的元组args。然后，print函数使用format方法将这些值格式化为一个带有索引和列值的字符串，打印出来。
这里的'{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'是一个格式化字符串，用于指定输出的格式。具体解释如下：
{:2}: 表示将第一个参数i格式化为占据2个字符的字符串，如果不足两个字符则在左侧填充空格。
{:20}: 表示将第二个参数args[0]格式化为占据20个字符的字符串，如果不足20个字符则在右侧填充空格。
{:6.2f}: 表示将第三个至第五个参数args[1]至args[3]格式化为浮点数，其中6.2f表示总宽度为6，其中小数点后有两位小数。
在format方法中，i代表索引，*args用于展开元组args，因此args[0]到args[3]分别对应于bad_mean、mid_mean和good_mean中的值。
'''

total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6] # 提取所有的二氧化硫指数，尝试找出比平均值低的
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
# lt就是less than
predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum()
# (torch.Size([4898]), torch.bool, tensor(2727))
actual_indexes = target > 5
actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()
# (torch.Size([4898]), torch.bool, tensor(3258))
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
# .item()函数代表将tensor转化为scalar
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
n_matches, n_matches / n_predicted, n_matches / n_actual
# (2018, 0.74000733406674, 0.6193984039287906) 后面两个一个是precision, 一个是recall
# precision：模型找出来的里面多少是正确的？ recall:机器找出来的正确的占真正正确的的比率
# 这里我们尝试的是“能不能用二氧化硫是否高于平均值来判断是不是好酒
````

接下来学习时间序列相关的操作

````python
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)

bikes_numpy = np.loadtxt(
    r"C:\Users\23714\Desktop\SKD时期文件\4VDLab\dlwpt-code-master\data\p1ch4\bike-sharing-dataset\hour-fixed.csv", 
    dtype=np.float32, 
    delimiter=",", 
    skiprows=1, 
    converters={1: lambda x: float(x[8:10])}) # <1>
bikes = torch.from_numpy(bikes_numpy)
'''
在这段代码中，converters参数是np.loadtxt函数的一个参数，用于指定如何转换文件中的某些列数据。具体来说，converters是一个字典，其中键表示要转换的列的索引，值是一个函数，用于将原始数据转换为需要的格式。在这里，converters={1: lambda x: float(x[8:10])}指定要将索引为1的列进行转换，转换方式是将每个元素的第8到第9个字符（不包括第9个字符）提取出来，然后将提取出来的字符串转换为浮点数。
Converts data strings to numbers corresponding to the day of the month in column 1
'''
bikes.shape, bikes.stride() # (torch.Size([17520, 17]), (17, 1))
daily_bikes = bikes.view(-1, 24, bikes.shape[1]) # -1就代表730了，因为代表计算机自动计算这里是多少
daily_bikes.shape, daily_bikes.stride()
# (torch.Size([730, 24, 17]), (408, 17, 1))
daily_bikes = daily_bikes.transpose(1, 2) # 转换一下维度
daily_bikes.shape, daily_bikes.stride()
# (torch.Size([730, 17, 24]), (408, 1, 17))
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
first_day[:,9]
# tensor([1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2])
weather_onehot.scatter_(
    dim=1, 
    index=first_day[:,9].unsqueeze(1).long() - 1, 
    value=1.0)
torch.cat((bikes[:24], weather_onehot), 1)[:1]

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4,
                                   daily_bikes.shape[2])
daily_weather_onehot.shape # torch.Size([730, 4, 24])
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min))
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp)) / torch.std(temp))
````

表示文本

````python
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)

with open(r"C:\Users\23714\Desktop\SKD时期文件\4VDLab\dlwpt-code-master\data\p1ch4\jane-austen\1342-0.txt", encoding='utf8') as f:
    text = f.read()
lines = text.split('\n')
line = lines[200]
# 尝试对这一行进行独热编码
letter_t = torch.zeros(len(line), 128) # <1> 
letter_t.shape # torch.Size([70, 128])
for i, letter in enumerate(line.lower().strip()):
# .lower()是变成小写，strip()是为了删除不必要的空格
    letter_index = ord(letter) if ord(letter) < 128 else 0  # <1>
# ord()返回的就是Unicode码
    letter_t[i][letter_index] = 1

def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    # 用空格取代掉换行符，然后split以空格为划分，因为习惯上是逗号后面一个空格！但是逗号之类的会进入列表
    word_list = [word.strip(punctuation) for word in word_list]
    # 接下来每一个单次元素的标点符号都删掉
    return word_list

words_in_line = clean_words(line)
line, words_in_line
'''
('“Impossible, Mr. Bennet, impossible, when I am not acquainted with him',
 ['impossible','mr','bennet','impossible','when','i','am','not','acquainted','with','him'])
'''
word_list = sorted(set(clean_words(text)))
# 对文本提取元素，然后set()转化为集合，因此可以排除掉相同的元素
word2index_dict = {word: i for (i, word) in enumerate(word_list)}
# enumerate(注意enumerate返回的是什么)遍历word_list中的每一个集合，然后变成字典
len(word2index_dict), word2index_dict['impossible'] # (7261, 3394)

word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    # 这一步可以为onehot做铺垫，就是对应编号的index的位置变成1，代表这一单词是这一行的1的index所代表的word
    print('{:2} {:4} {}'.format(i, word_index, word))
    
print(word_t.shape)
'''
 0 3394 impossible
 1 4305 mr
 2  813 bennet
 3 3394 impossible
 4 7078 when
 5 3315 i
 6  415 am
 7 4436 not
 8  239 acquainted
 9 7148 with
10 3215 him
torch.Size([11, 7261])
'''
word_t = word_t.unsqueeze(1)
word_t.shape # torch.Size([11, 1, 7261])

[(c, ord(c)) for c in sorted(set(text))] # 查看Unicode的编码规则
ord('1') # 返回1的Unicode代码
````

## Chapter 5

从这里开始，要真正实现logistic regression的细节了。

下面是一个实现简单线性回归的例子

````python
%matplotlib inline
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, linewidth=75)
import matplotlib.pyplot as plt

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
# 上面是得到的数据，下面转化为tensor
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
# 用matplotlib方法完成绘图
fig = plt.figure(dpi = 400)
plt.xlabel("Measurement")
plt.ylabel("Temperature(celcius)")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
# 这里还是完成简单的线性回归
def model(t_u, w, b):
    return w * t_u + b
def loss_fn(t_p, t_c): # t_p : predicted
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()
# 初始化；当然随机初始化也可以
w = torch.ones(())
b = torch.zeros(())
# 小括号 () 表示一个空的元组。这个空元组作为参数传递给torch.ones函数，告诉函数创建一个形状为空的张量，即一个 # 标量（scalar）张量。标量张量是只包含一个元素的张量，没有维度
t_p = model(t_u, w, b)
loss = loss_fn(t_p, t_c)
delta = 0.1
# 接下来要应用gradients descent
# 这里求导数的方法利用的是微小扰动法
loss_rate_of_change_w = 
    (loss_fn(model(t_u, w + delta, b), t_c) - 
     loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)
loss_rate_of_change_b = 
    (loss_fn(model(t_u, w, b + delta), t_c) - 
     loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)
# 用学习率更新w and b
learning_rate = 1e-2
w = w - learning_rate * loss_rate_of_change_w
b = b - learning_rate * loss_rate_of_change_b
# 下面就是真求导了，为什么那么多导数？因为链式法则
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)  # <1>
    return dsq_diffs
def dmodel_dw(t_u, w, b):
    return t_u
def dmodel_db(t_u, w, b):
    return 1.0
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])
# 注意返回的是总的dJ因此有sum()

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
		# params是一个二维的东西
        t_p = model(t_u, w, b)  
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)  # <2>
		# 看下面一个式子，也就知道了为什么grad_fn这个返回值要stack了
        params = params - learning_rate * grad
		# 完成了参数的更新，并且准备投身下一个epoch的训练
        print('Epoch %d, Loss %f' % (epoch, float(loss))) # <3>
            
    return params
# 更详细的版本
def training_loop(n_epochs, learning_rate, params, t_u, t_c,
                  print_params=True):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)  # <1>
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)  # <2>

        params = params - learning_rate * grad

        if epoch in {1, 2, 3, 10, 11, 99, 100, 4000, 5000}:  # <3>
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params:', params)
                print('    Grad:  ', grad)
        if epoch in {4, 12, 101}:
            print('...')

        if not torch.isfinite(loss).all():
            break 
            
    return params
'''
torch.isfinite(loss)会返回一个布尔张量，其中每个元素表示对应位置的损失是否有限（即不是无穷大或者NaN）。
.all()函数用于检查张量中的所有元素是否都为True，如果是，则返回True；否则返回False。
因此，if not torch.isfinite(loss).all()的意思是，如果损失值中有任何一个元素是无穷大或者NaN，那么条件成立，执行相应的操作。
'''

training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2, 
    params = torch.tensor([1.0, 0.0]), 
    t_u = t_u, 
    t_c = t_c)
'''
shapes: x: torch.Size([]), y: torch.Size([3, 1])
        z: torch.Size([1, 3]), a: torch.Size([2, 1, 1])
x * y: torch.Size([3, 1])
y * z: torch.Size([3, 3])
y * z * a: torch.Size([2, 3, 3])
tensor(1763.8848)
tensor([35.7000, 55.9000, 58.2000, 81.9000, 56.3000, 48.9000, 33.9000,
        21.8000, 48.4000, 60.4000, 68.4000])
Epoch 1, Loss 1763.884766
    Params: tensor([-44.1730,  -0.8260])
    Grad:   tensor([4517.2964,   82.6000])
Epoch 2, Loss 5802484.500000
    Params: tensor([2568.4011,   45.1637])
    Grad:   tensor([-261257.4062,   -4598.9702])
Epoch 3, Loss 19408029696.000000
    Params: tensor([-148527.7344,   -2616.3931])
    Grad:   tensor([15109614.0000,   266155.6875])
...
Epoch 10, Loss 90901105189019073810297959556841472.000000
    Params: tensor([3.2144e+17, 5.6621e+15])
    Grad:   tensor([-3.2700e+19, -5.7600e+17])
Epoch 11, Loss inf
    Params: tensor([-1.8590e+19, -3.2746e+17])
    Grad:   tensor([1.8912e+21, 3.3313e+19])
因为我们学习率给的过大，因此发生了梯度爆炸，最终无法收敛
'''
training_loop(
    n_epochs = 100, 
    learning_rate = 1e-4, 
    params = torch.tensor([1.0, 0.0]), 
    t_u = t_u, 
    t_c = t_c)
'''
Epoch 1, Loss 1763.884766
    Params: tensor([ 0.5483, -0.0083])
    Grad:   tensor([4517.2964,   82.6000])
Epoch 2, Loss 323.090515
    Params: tensor([ 0.3623, -0.0118])
    Grad:   tensor([1859.5493,   35.7843])
Epoch 3, Loss 78.929634
    Params: tensor([ 0.2858, -0.0135])
    Grad:   tensor([765.4666,  16.5122])
...
Epoch 10, Loss 29.105247
    Params: tensor([ 0.2324, -0.0166])
    Grad:   tensor([1.4803, 3.0544])
Epoch 11, Loss 29.104168
    Params: tensor([ 0.2323, -0.0169])
    Grad:   tensor([0.5781, 3.0384])
...
Epoch 99, Loss 29.023582
    Params: tensor([ 0.2327, -0.0435])
    Grad:   tensor([-0.0533,  3.0226])
Epoch 100, Loss 29.022667
    Params: tensor([ 0.2327, -0.0438])
    Grad:   tensor([-0.0532,  3.0226])
    
    tensor([ 0.2327, -0.0438])
'''
# 那么其实还有其他防止梯度爆炸的方法
t_un = 0.1 * t_u # 有点归一化的味道，让y轴和x轴的高斯分布尽可能接近
training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2, 
    params = torch.tensor([1.0, 0.0]), 
    t_u = t_un, # <1>
    t_c = t_c)
params = training_loop(
    n_epochs = 5000, 
    learning_rate = 1e-2, 
    params = torch.tensor([1.0, 0.0]), 
    t_u = t_un, 
    t_c = t_c,
    print_params = False)
'''
Epoch 1, Loss 80.364342
Epoch 2, Loss 37.574913
Epoch 3, Loss 30.871077
...
Epoch 10, Loss 29.030489
Epoch 11, Loss 28.941877
...
Epoch 99, Loss 22.214186
Epoch 100, Loss 22.148710
...
Epoch 4000, Loss 2.927680
Epoch 5000, Loss 2.927648
tensor([  5.3671, -17.3012])
'''
````

下面学习的autogrid and optimizer是以后真正常用的

````python
%matplotlib inline
import numpy as np
import torch
torch.set_printoptions(edgeitems=2)

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
                    3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                    33.9, 21.8, 48.4, 60.4, 68.4])
t_un = 0.1 * t_u
def model(t_u, w, b):
    return w * t_u + b
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()
params = torch.tensor([1.0, 0.0], requires_grad=True)
# 这个张量被设置为需要梯度计算，即requires_grad=True，这意味着在张量上进行的操作将被跟踪，以便在反向传播过程中计算梯度。
# 在深度学习中，通常将这样的张量用作模型的参数，并且在训练过程中，会根据损失函数计算的梯度来更新这些参数，以使模型逐渐优化到最佳状态。
params.grad is None # True, 代表尚未进行计算
loss = loss_fn(model(t_u, *params), t_c)
# 星号代表这个数组展开成两个变量
loss.backward() # 反向传播一次
# PyTorch 的自动求导机制会自动跟踪张量上的操作，并构建计算图以便计算梯度
params.grad # tensor([4517.2969,   82.6000])
# 下面是重置梯度
if params.grad is not None:
    params.grad.zero_()
    
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        # 重置梯度
        if params.grad is not None:  
            params.grad.zero_()
        
        t_p = model(t_u, *params) 
        loss = loss_fn(t_p, t_c)
        loss.backward() # 这一步会自动计算梯度
        
        with torch.no_grad():  
            params -= learning_rate * params.grad
		# 参数更新
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return params
training_loop(
    n_epochs = 5000, 
    learning_rate = 1e-2,  # 注意我已经“归一化”过了
    params = torch.tensor([1.0, 0.0], requires_grad=True), # <1> 
    t_u = t_un,  
    t_c = t_c)
'''
tensor([1., 0.], requires_grad=True)
True
Epoch 500, Loss 7.860115
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957698
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927679
Epoch 4500, Loss 2.927652
Epoch 5000, Loss 2.927647
tensor([  5.3671, -17.3012], requires_grad=True)
'''
````

以上就是autogrid的手法了，尤其是training_loop函数，已经大大帮忙减轻了代码的苦恼。

下面学习optimizer：

````python
# 这一部分紧接着上文
import torch.optim as optim
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)
# Stochastic Gradients Descent; 要优化的params一定要放在里面
# 分步骤看：
t_p = model(t_u, *params) # 通过这里传入params, 来实现loss.backward()对params的控制
loss = loss_fn(t_p, t_c)
loss.backward() # 会自动计算梯度值，就是params.grid
optimizer.step() # 会自动根据params.grid去更新params
# 等效于： params -= learning_rate * params.grid

def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params) 
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad() # 将梯度清零
        loss.backward() # 同时会计算params.grid
        optimizer.step() # 会自动根据params.grid去更新params

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return params
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate) # <1>

training_loop(
    n_epochs = 5000, 
    optimizer = optimizer,
    params = params, # <1> 
    t_u = t_un, # 注意，传入的是un
    t_c = t_c)
'''
Epoch 500, Loss 7.860115
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957698
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927679
Epoch 4500, Loss 2.927652
Epoch 5000, Loss 2.927647
tensor([  5.3671, -17.3012], requires_grad=True)
'''
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) # <1>

training_loop(
    n_epochs = 2000, 
    optimizer = optimizer,
    params = params,
    t_u = t_u, # <2> 
    t_c = t_c)
'''
Epoch 500, Loss 7.612898
Epoch 1000, Loss 3.086700
Epoch 1500, Loss 2.928579
Epoch 2000, Loss 2.927644
tensor([  0.5367, -17.3021], requires_grad=True)
'''
# 注意，上面传入的是t_u；但是在这么大的步长下，为什么依然没有发散呢？那是因为我们使用的是Adam
# Auto adaptive optimizer, 帮助我们实现了normalization
# 要是SGD，500轮早就梯度爆炸了

# 一般在真正训练的过程中，通常是要抽出百分之二十的数据来做验证集
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

# 由于是SGD, 还是乘以0.1比较好
train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u,
                  train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params) # <1>
        train_loss = loss_fn(train_t_p, train_t_c)
                             
        val_t_p = model(val_t_u, *params) # <1>
        val_loss = loss_fn(val_t_p, val_t_c)
        
        optimizer.zero_grad()
        train_loss.backward() # <2>
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")
            
    return params
````

























