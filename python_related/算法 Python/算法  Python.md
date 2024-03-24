# 算法  Python

# 时间复杂度和空间复杂度

##  估计算法运行效率与时间复杂度

````python 
print('Hello World')
for i in range(n):
    print('Hello World')
for i in range(n):
    for j in range(n):
        print('Hello World')
````

上面三组代码,哪组运行时间最短?

可能我会知道肯定是第一个最快,但是每个代码运行的时间是多少,我们能知道吗

换句话说,用什么方式来体现算法运行的快慢?

可能我们会用时间来衡量,但是有缺陷:

1. 用来进行运算的及其可能不同
2. 机器的运行速度可能不同
3. n的大小是不同的! 我应该选什么n来测算时间?

因此引入时间复杂度来衡量算法的快慢! 它是用来评估算法运行的一个式子

则第一行, 我们用O(1)来表示    O表示大约,**而1代表的是单位**

1并不是指1秒或者1毫秒什么的,就是一种单位

而第二行就是O(n), 循环会运行n次; 那么第三行就是O(n^2^)

````python 
print('Hello World')
print('Hello World')
print('Hello World')

for n in range(n):
    print('Hello World')
    for j in range(n):
        print('Hello World')
````



**第一块的代码, 应该是O(3); 而第二块的代码**, **应该是O(n^2^+n)吗???**

**很可惜,并不是这样的!!!**     n^2^是单位, 1 是单位, n 也可能是单位,但是3不能是单位!

惊奇的是,第一块代码应该是 O(1)  !!!

为什么呢?  在计算机中, 基本操作只要是不上升到n规模,它的运算就是1个单位!!

或者我们换句话说, 我们现在只是估计计算效率

但是为什么n^2^+n不正确呢?  n^2^是一个单位,n也是,那我们只保留n^2^,即 O(n^2^)

````Python
while n > 1:
    print(n)
    n = n//2
````

那么上面的第一个代码,其运算时间复杂度呢?

输入64,会输出64  32  16  8  4  2

注意这里是//2, 而不是-2; 我们知道2^6^=64, 则我们表示:

O (log~2~n)   或者 O(logn)   只有次次折半, 才会出现logn

时间复杂度小结:   常见时间复杂度(按照效率排序)

1 < logn < n < nlogn < n^2^ < n^2^logn < n^3^

## 简单判断算法复杂度

首先我们确定规模n, 如果是循环减半 logn ,如果是k层关于n的循环n^k^
如果情况复杂,则根据具体问题进行分析

## 空间复杂度

空间复杂度是用来评估算法内存占用大小的式子,它的表现形式和时间复杂度完全一样

算法使用了几个变量: O(1)   算法使用了长度为n的一维列表: O(n)

m行n列呢   O(mn)

有句话说的很有意思:  空间换时间   这是算法中青睐的法则

我宁可多占空间,也想把时间削减下来, 例如分布式运算

或者更直观的例子, 一个代码可以在一台设备上跑, 但是为了速度, 我就可以把代码拆成多个部分,给多个设备分别跑

# 递归

首先递归有两个特点: 调用自身, 和结束条件

````Python
def func1(x):
    if x>0:
        print(x)
        func1(x-1)
def func2(x):
    if x>0:
        print(x)
        func2(x+1)
def func3(x):
    if x>0:
        func(x-1)
        print(x)
````

上面有几个定义的函数

对于func1来说: 先是打印,然后函数再递归

func3呢? 先递归,后打印

这两个东西看起来十分接近啊?但是其实差别非常大

加入我们输入的是3,那么func1会输出3 2 1

但是func3 会输出1 2 3

为什么呢?  有一个非常好的解释图

![image-20230819221528187](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230819221528187.png)

![image-20230819221555461](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230819221555461.png)

总之,这两个东西就是不一样的东西

应用: 汉诺塔问题      这是十分重要的经典!!!!

如果是n个圆盘, 请设计一个程序, 演示移动过程

如果是两个圆盘, 我们很容易知道怎么移动

但是如果是n个呢? n-1 : A-C-B   最底下的: A-C  n-1: B-A-C

**我们将最底下的一个视为一个,上面的n-1视为1个**

这样我们想让最底下的移动到目标地点,就和n=2 的情况是一样的了

````Python
def hanoi(n,a,b,c):# a b c代表从a到b到c,说白了,这个函数就是一系列移动
    if n>0: # 这是终止条件!如果没有盘,那根本动都不用动!
        hanoi(n-1,a,c,b)
        print('Moving from %s to %s' % (a,c))
        hanoi(n-1,b,a,c) 
````

当然递推式子是很好求的,但是具体的过程还是十分神奇的

其实,我们把最上面的n-1个a-c-b把最底下的放在c,然后n-1个b-a-c,就完成了

那么n-1  a-c-b  b-a-c分别是怎么完成的呢? 他们的移动也是hanoi函数完成的,但是注意我要换一下位置参数的移动顺序

# 查找

## 列表查找  LINEAR SEARCH

查找:在一些数据元素中,通过一定方法找出与给定关键词相同的数据元素的过程

例如:index()

列表查找:在列表中查找指定元素,输入列表和代查找元素,然后输出元素下标

顺序查找(linear search)  从第一个元素开始,按照顺序进行搜查,知道找到第一个目标元素为止

例如:

````python
for index , v in enumerate(list):
    if v == val:
        print(index)
else:
    print('No!!')
````

注意enumerate函数返回的是包含索引和对应元素的元组,这两个东西会分别赋值给index和v

且for  else语句是正确的!!!  注意,这个else后面的语句是一定要在循环没有被打断的情况下才会执行的!!

## 二分查找  BINARY SEARCH

又叫折半查找,从有序列表的初始候选区list[0:n]开始,通过对待查找的值与候选区中间值得比较,可以使候选区减少一半

事例:  1 2 3 4 5 6 7 8 9    我想找到3   left = 1   right = 9   mid = 5

5 > 3   则   left = 1   right = 4    mid = (1+4) //2 = 2   2<3

则  left = 3  right = 4  终于找到了

代码实现:

````python
def binary_search(li, val):
    left = 0
    right = len(li)-1
    while left <= right:   # 候选区里面有值
        mid = (left + right)//2
        if li[mid] == val:
            return mid
        elif li[mid] > val:
            right = mid-1
        else:
            left = mid+1
    else:
        return None
li = [1,2,3,4,5,6,7,8,9]
binary_search(li,3)    
````

那么我们可以看出来,二分查找的时间复杂度就是O(logn),效率更高! 肯定比线性查找好

但是,如果列表是无序的,那么只能线性查找了

# 排序

## 列表排序

内置的排序函数sort(), 对列表进行升序排序

注意  sort是直接在原列表上面操刀,而sorted是创建一个新列表然后排序,原列表是不变的

常见的排序算法:

low: 冒泡排序  选择排序  插入排序

nb! : 快速排序   堆排序  归并排序

其他排序:  希尔排序  基数排序  基数排序

## 冒泡排序  BUBBLE SORT

基本思想:  列表中每两个相邻的数,如果后面的比前面打,则交换两个数,以进行升序

一趟之后, 无序区域减少一个元素,有序区域增加一个元素

这个代码的关键点: 趟  无序区域范围

代码实现:

````python
import random
def bubble_sort(li):
    for i in range(len(li)-1):    # i表示第i趟
        for j in range(len(li)-i-1):
            if li[j]>li[j+1]:
                li[j],li[j+1] = li[j+1],li[j]
li = [random.randint(0,10000) for i in range(1000)]
bubble_sort(li)
````

为什么趟是十分关键的元素? 因为每一趟都会将最大的数放到最后面, 创造有序区域

有序区域元素个数为列表元素个数的时候,那就是完成了排序

这个算法的时间复杂度是 O(n^2^),但是,如果有一时刻列表已经排好序了呢?

这样就没有必要再排一次序了,不然的话根据for语句,我们一定走满n趟

因此进行补充和改进

````python
import random
def bubble_sort(li):
    for i in range(len(li)-1):    # i表示第i趟
        exchange = False
        for j in range(len(li)-i-1):
            if li[j]>li[j+1]:
                li[j],li[j+1] = li[j+1],li[j]
                exchange = True
        if not exchange:
            return

````

每一趟之后检查:如果这一趟进行了交换,那么exchange = True,在后面的检验中不通过,再排序

那么如果这一趟啥也没有换,exchange = False, 那么在后面的检验中会立刻停止,用return打断

## 选择排序  SELECT SORT

这十分简单,直接上代码

````python
def select_sort(li):
    li_new = []
    for i in range(len(li)):
        min_val = min(li)
        li_new.append(min_val)
        li.remove(min_val)
    return li_new
````

## 插入排序  INSERT SORT

把他比作打牌:牌无序,我抽出一张牌,然后放进手牌,没抽出一张,放在紧接着比他大的牌的前面

````python
def insert_sort(li):
    for i in (1,len(li)):   # 表示摸到的牌的下标
        tmp = li[i]
        j = i - 1   #  j指的是手里的牌的下标
        while li[j] > tmp and j >= 0:   #  找到我最后放牌的位置
            li[j+1] = li[j]   #  移动牌
            j -= 1
        li[j+1] = tmp        
````

它的时间复杂度是O(n^2^)

## 快速排序  QUICK SORT

思路: 取第一个元素,使元素P归位;列表被分成两部分,左边都比p小,右边都比p大,然后递归完成排序

````python
def partition(data,left,right):
    tmp = li[left]
    while left < right:
        while left<right and li[right] >= tmp: # 找比tmp小的数
            right -= 1  
        li[left]=li[right]
        while left<right and li[left] <= tmp:
            left += 1
        li[right]=li[left]
    li[left] = tmp
    return left   #  归位的位置的下标        
def quick_sort(data,left,right):
    if left < right:   #  代表至少有两个元素
        mid = partition(data,left,right)
        quick_sort(data,left,mid-1)
        quick_sort(data,mid+1,right)
# example
li = [5,7,4,6,3,1,2,9,8]
partition(li,0,len(li)-1)
quick_sort(li,0,len(l1)-1)
````

那么p归位是什么呢:只可意会不可言传,见代码中的partition()

它的时间复杂度是O(nlogn),当然,它的最坏复杂度是O(n^2^)

## 堆排序  HEAP SORT

### 前置知识:1. 树与二叉树

树:

树是一种数据结构,比如目录结构;而树是一种可以递归定义的数据结构

树是n个节点组成的集合,如果n=0,那么就是一颗空树

如果n > 0, 那么存在一个节点作为树的根节点,其他节点可以分为m个集合,每个集合本身又是一棵树

介绍一些概念: 根节点  叶子节点  (不能再分支的点)  树的深度(高度)  树的度  孩子节点/父节点  子树

什么是度?分几个差,就说这个点的度是多少; 而树的度就是整棵树里面最大的度

而二叉树就是度最大为2的数: 就是说,每个点最多两个孩子节点, 分为左孩子节点和右孩子节点

满二叉树: 一个二叉树,如果每一层的节点数都达到最大值,则这个二叉树就是满二叉树

完全二叉树: 叶节点只能出现在最下层和次下层,并且最下面一层的节点都集中在该层最左边的若干位置的二叉树

其实可以这么看:完全二叉树由满二叉树变过来,从最后一个节点开始,**按照顺序的逆顺序**开始删除叶节点!

### 2. 二叉树的存储方式

二叉树的存储方式(表达方式)有两种: 链式存储方式和**顺序存储方式**

这个顺序存储方式和列表紧密相关!   

父节点和左孩子节点的编号小标有什么关系?  i --- 2i + 1

父节点和右孩子节点的编号下标有什么关系?   i --- 2i + 2

### 堆排序  1. 什么是堆和堆的向下调整性质

堆是一种特殊的完全二叉树,分为大根堆和小根堆

大根堆:一棵完全二叉树,满足任意节点都比其他孩子节点大

那么小根堆相反: 任意节点都比孩子节点小

性质:当根节点的左右子树都是堆的时,可以通过一次向下的调整来将其变换成一个堆(但是整个不是堆)

### 2. 堆排序的过程

步骤: 

1. 建立一个堆   2. 得到堆顶元素,为最大元素    3. 去掉堆顶,将堆的最后一个元素放到堆顶(是为了保证变化之后这个东西还是一个完全二叉树).此时可以通过一次调整重新使堆有序(把第二大元素放在顶部)

4. 堆顶元素为第二大元素   5. 重复步骤3, 直到堆变为0

那么如何建立堆呢? 从底下开始 , "农村包围城市"

注意,我们最好不要额外设计一个空列表去装这些一次出来的元素呢? 

因此我们设计: 输出元素放在最后

````python
def sift(li,low,high)  # low high分别代表堆的第一个和最后一个元素的下标
    i = low # 父节点那一层,最开始是在最顶层,指向根节点
    j = 2 * i + 1  # j开始是左孩子
    tmp = li[low]  # 把堆顶存起来
    while j <= high:   # j位置有节点!
        if j + 1 <= high and li[j+1] > li[j]:  # 要有右函数且右孩子比左孩子大
            j = j + 1  #  j指向右孩子!
        if li[j] > tmp:
            li[i] = li[j]
            i = j               #  往下看一层
            j = 2 * i + 1
        else:  #  tmp更大,把tmp放到i的位置上
            li[i] = tmp
            break
    else:
        li[i] = tmp   #  把tmp放在最后一排的叶子结点上

def heap_sort(li):
    n = len(li)
    for i in range((n-2)//2,-1,-1):  #  i代表建队的时候的小堆的根节点的下标
    '''
    注意!我们想要从最后的非叶节点开始对每个小堆进行操作
    对于最后一个叶节点的爹,它的下标一定是(n-2)//2,而且无论是左还是右
    从这个非叶节点开始向前开始一个一个小单位去进行sift
    '''
        sift(li,i,n-1)  #  high这里让它取n-1是完全没问题的!high存在的意义仅仅是
    #  判断i的左右子节点是否存在
    print(li)  #  建堆建完了
    for i in range(n-1,-1,-1):  #  i指向当前最后一个元素
        li[0],li[i] = li[i],li[0]
        sift(li,0,i-1)  # 注意最后一个会变成i-1!!把最顶上的放在最后,我们不让它再走sift
    print(li)  #  堆排序完成            
````

注意这个函数没有用到递归!

###  3. 堆排序在python中的内置函数

````python
import heapq
import random
li = list(range(100))
random.shuffle(li)
print(li)
heapq.heapify(li)  #  建堆
heapq.heappop(li)  #  每次打印一个最小的数字
````

### 4. 堆排序------topk问题

有n个数,设法得到前k大的数,如果先排序后切片,那么就会非常慢

我们可以heap k次,我们就能得到前十个数字!

这样,复杂度就是 O(nlogk)   ( < O(nlogn) )

实现:

````python
def sift(li,low,high)  # low high分别代表堆的第一个和最后一个元素的下标
    i = low # 父节点那一层,最开始是在最顶层,指向根节点
    j = 2 * i + 1  # j开始是左孩子
    tmp = li[low]  # 把堆顶存起来
    while j <= high:   # j位置有节点!
        if j + 1 <= high and li[j+1] < li[j]:  # 要有右函数且右孩子比左孩子大
            j = j + 1  #  j指向右孩子!
        if li[j] < tmp:
            li[i] = li[j]
            i = j               #  往下看一层
            j = 2 * i + 1
        else:  #  tmp更小,把tmp放到i的位置上
            li[i] = tmp
            break
    else:
        li[i] = tmp   #  把tmp放在最后一排的叶子结点上
def topk(li,k):
    heap = li[0:k]
    for i in range((k-2)//2,-1,-1)
        sift(heap,i,k-1)
    for i in range(k,len(li)):
        if li[i] > heap[0]:
            heap[0] = li[i]
            sift(heap,0,k-1)
    for i in range(n-1,-1,-1):  #  i指向当前最后一个元素
        heap[0],heap[i] = heap[i],heap[0]
        sift(li,0,i-1) 
    return heap
````

可以看出,解决思路是:

1. 取列表前k个元素建立小根堆,堆顶就是目前第k大的数
2. 依次向后遍历原列表,对于列表中的元素,如果小于堆顶,则忽略该元素; 如果大于堆顶,则将堆顶更换为该元素,并且对堆进行一次调整
3. 遍历列表所有元素后,倒序弹出堆顶

最大的亮点就是,每一次遍历(n)后,heap的时间复杂度是logk 而不是logn

注意:为什么要小根堆? 因为对于每一次取出的原列表中的元素,我要的是它比堆中最小的元素还小!

最后就能真正保证堆里面的数字是前k大的

## 归并排序 MERGE SORT

### 1. 归并 MERGE

假设现在列表分为两段有序, 如何将其合成一个有序列表?

2 5 7 8 9     1 3 4 6   前面五个数是有序的, 后面四个数是有序的

指针分别放在首位, 1出列,  右指针放在了3; 再后面2和3比, 2出列,以此类推

````python
def merge(li,low,mid,high):   
    i = low
    j = mid + 1
    ltmp = []
    while i <= mid and j <= high  #  只要左右两边都有数
        if li[i] < li[j]:
            ltmp.append(li[i])
            i += 1 
        else:
        	ltmp.append(li[j])
            j += 1
    # while执行完了,肯定有一部分没有数字了
    while i <= mid:
    	ltmp.append(li[i])
    	i += 1
    while:
        ltmp.append(li[j])
        j += 1
    li[low:high+1] = ltmp  #  为什么不是0? 因为接下来要设计递归        
````

### 2. 使用归并

分解: 将列表越分越小,直至分成一个元素

终止条件: 一个元素是有序的

合并: 将两个有序列表归并,列表越来越大

````python
def merge_sort(li,low,high):
    if low < high: #  至少有两个元素,递归
        mid = (low +high ) // 2
        merge_sort(li,low,mid)
        merge_sort(li,mid+1,high)
        merge(li,low,mid,high)
````

从递归的思想看这个函数: 我核心是只想左右两边都完成归并(对应3,4,5行), 然后最后两大块进行归并(6),就成功了那么这两大块是如何完成归并的呢? 其实利用递归, 这两大块自己内部的调整其实和原来的函数所是实现的调整是一样的!! 因此这个代码是完全正确的! 只要确定终止条件是: 最后分的最小的单元是必须有至少两个元素! 

### 3. 归并排序的时间复杂度是多少?

这种方法最大的特点是: 不是原地排序! 事实上, python中的sort事实上结合的是归并和插入

它的时间复杂度是O(nlogn)

## NB三人组总结:

运行时间: 快速 < 归并 < 堆

在时间复杂度相同的状况下, 三种排序各有缺点:

快速排序: 在极端情况下, 效率非常低

归并排序: 需要额外的内存开销

堆排序: 在快的排序算法中算相对较慢的

![image-20230818151814159](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230818151814159.png)

注意: 快速排序的空间复杂度并不是O(1), 为什么呢? 因为递归是需要使用内存的!! 系统需要记忆每一次递归上一次的状态, 这其实是很容易理解的

那么: 稳定性又是指什么呢? 在现实中,我们的数据不可能仅仅是数字

````python 
{'name':'a','age':18}
{'name':'a','age':25}
{'name':'b','age':20}
````

如果是稳定的排序, 那么排完后顺序不变, 因为a > b; 但是如果是不稳定的, 那么就只能按照18 20 25来排

有顺序的挨个换的, 那么就是稳定的; 如果是飞来飞去的比的, 那么就不是稳定的

## 希尔排序 SHELL SORT

它是一种分组插入排序算法

首先取一个整数d~1~=n/2, 将元素分成这么多组,每组相邻元素之间的距离为d~1~ , 在各组内进行直接插入排序

取第二个整数d~2~=d~1~/2, 重复上述步骤,知道d~i~ = 1, 即所有元素在同一组内进行直接插入排序

注意: 希尔排序每趟并不是使某些元素有序,而是使整体数据越来越接近有序; 最后一趟使所有数据有序

````python 
def insert_sort_gap(li,gap):
    for i in range(gap,len(li)):
        tmp = li[i]
        j = i-gap
        while j >= 0 and li[j] > tmp:
            li[j+gap]=li[j]
            j -= gap
        li[j+gap] = tmp
````

事实上, 原来的插入排序就是gap = 1 , 因此在源代码上面直接改动即可

````python
def shell_sort(li):
    d = len(li)//2
    while d >= 1:
        insert_sort_gap(li,d)
        d //= 2
````

实验结果显示: 这种排序比插入排序快很多, 但是还是比NB三人组里面的慢

![image-20230818155608425](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230818155608425.png)

根据上图显示, 这种希尔排序当你选的标准(gap)不一样, 对应的时间复杂度也是不一样的

目前来看, 最快的是A003586

## 计数排序 COUNT SORT

一种对列表进行排序, 已知列表中的数范围在0-100之间, 设计时间复杂度为O(n)的算法

````:
def count_sort(li,max_count=100):
    count = [0 for _ in range(max_count+1)]
    for val in li:
        count[val] += 1 
    li.clear()
    for ind , val in enumerate(count):
        for i in range(val):
            li.append(ind)
````

这个排序的原理十分简单, 通过代码就可以看出来我们想干什么

精妙的地方就在于, 如果0-100 中这个数字根本没出现, 对应次数为0, 在for循环就自动不会进行! 

而删除原列表而在原列表中继续写, 是为了减少内存的占据; 但是count列表的存在客观上会占内存

计数排序的复杂度为O(n), 但是仍然有缺点, 引出了下面的桶排序

## 桶排序 BUCKET SORT

在计数排序中, 如果元素的范围较大(1-1亿之间), 如何改进算法(count列表长度太大! )

那么首先将元素分在不同的桶中, 再度以每个桶中的元素排序

```` python
def bucket_sort(li,n = 100, max_num = 10000): # n为分的木桶的数量
    buckets = [[] for _ in range(n)]
    for var in li:
        i = min(var // (max_num // n),n-1)  # i表示var放在几号桶里面
        buckets[i].append(var)
        #  接下来我想让每个桶之间的元素有序
        #  这里使用了冒泡排序在桶内进行排序
        for j in range(len(buckets[i])-1,0,-1)  #  是倒过来了一个一个交换的
            if buckets[i][j] < buckets[i][j-1]:
                buckets[i][j],buckets[i][j-1]=buckets[i][j-1],buckets[i][j]
            else:
                break
    sorted_li = []
    for buc in buckets:
        sorted_li.extend(buc)
    return sorted_li
````

## 基数排序 RADIX SORT

前言: 假如说现在有一个员工表, 要求按照薪资进行排序, 在相同薪资的情况下按照年龄进行排序. 这很明显是一个多关键词排序

那么现在, 加入对32, 13 ,94, 52, 17, 54, 93 排序, 那么可不可以视为多关键词排序?

但是这里有一个很关键的点: 我们要的是稳定的排序; 就是在单一关键词下等效的两个不同元素, 能不能再排序之后依然保持正常的相对位置关系

那么如是观之, 放在薪资表问题当中, 我们可以先按照年龄进行排序, 然后再按照薪资进行稳定的排序; 回到这些数字的排序, 我们当然可以视为多关键词排序, 因为这些两位数都拥有十位和个位

我们可以先按照个位的数字进行分桶, 然后依次输出, 完成按照个位数进行的排序; 之后按照十位进行排序再次分桶, 最后依次输出, 保证了稳定的要求

当然如果这些数字位数不一样, 可以用0放在前面补齐, 最后再分桶的时候补入0桶; 在python中可以考虑对数进行找最高位数, 当然如果不想要用数学方法, 有其他方法

````python
def radix_sort(li):
    max_num = max(li) # 看最大数,确定最高位数
    it = 0
    while 10**it <= max_num: # 注意是<=
        buckets = [[] for _ in range(10)]
        for var in li:
            # 取出对应位数! 取模!
            digit = (var//(10**it))%10
            buckets[digit].append(var)
            # 完成该位的分桶
        li.clear()
        for buc in buckets:
            li.extend(buc)
        # 把数字重新写回li, 并且为稳定排序打下基础
        it += 1   # 成功找到最高位数
    # 当while完全结束, li最后被写回一次且保持稳定!
````

那么, 它的时间复杂度是多少呢?

时间复杂度: O(kn) 其中k代表循环次数(最大数的位数)

空间复杂度: O(k+n) 因为我们建立了bucket 序列

## 练习题目

### 例题1

给定两个字符串s和t, 判断t时候为s重新排列后组成的单词

思路: 如果s和t字母是一样的只是序列不一样, 则按照一定规则拍完序之后, 两个是一样的

````python
s = input()
t = input()
ss = list(s)
tt = list(t)
ss.sort()
tt.sort()
return ss == tt
````

这个代码的空间复杂度是O(nlogn), 因为我们使用了sort()

````python
return sorted(list(s)) == sorted(list(t))
````

另解: 用两个字典来保存这两个字符串保存的各个字母的数量

````python
dict1 = {}
dict2 = {}
for ch in s:
   dict1[ch] = dict1.get(ch,0) + 1 
   # 这行代码什么意思呢?
   # get会拿到上一个状态的chkey的value
   # 但是如果没有呢? 那么会赋予逗号右边的值
for ch in t:
   dict2[ch] = dict2.get(ch,0) + 1
print(dict1 == dict2)
````

从事实上看, 排序其实是更慢的

### 例题2

给定一个m*n的二维列表, 查找一个数书否存在, 列表有以下特征:

每一行的列表从左到右已经排列好; 每一行第一个数比上一行最后一个数大

````python
def search(matrix,target):
    for line in matrix:
        if target in line:
            return True
    return False
````

但是这个代码的运行复杂度非常大, 而且完全没有使用列表特性

````python
def search(matrix,target):
    list = []
    for line in matrix:
        list.extend(line)
    return target in list
````

这个就是稍微好一点, 依然没有使用列表特性

````python
def search(matrix,target):
    h = len(matrix)
    if h == 0:
        return False
    w = len(matrix[0])
    if w == 0:
        return False # 注意边界条件!!
    left = 0
    right = w*h - 1
    while left <= right:
        mid = (left+right)//2
        i = mid//w
        j = mid%w
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] > target:
            right = mid - 1
        else:
            left = mid + 1
    else:
        return False
````

这样就利用了有序的特性, 进行了二分查找, 快了很多

### 例题3

给定一个列表和一个整数, 设计算法找到两个数的下标, 使得两个数之和为给定的整数, 测试时保证肯定有一个结果, 用[x,y] 表示

````python
def twosum(nums,target):
    n = len(nums)
    for i in range(n):
    	for j in range(n):
            if  i!= j and nums[i] + num[j] == target:
            	return sorted[i,j]
````

注意这个代码最重要的是i和j不可以相同! 而且, 时间太慢!!

````python
def twosum(nums,target):
    n = len(nums)
    for i in range(n):
        for j in range(i):
            if nums[i] + num[j] == target:
                return sorted[i,j]
````

换个思路!!

````python
def twosum(nums,target):
    for n in (target//2+1):
        if n in nums and (target-n) in nums:
            return sorted[nums.index(n),nums.index(target-n)]
````

# 数据结构基础

## 数据结构

数据结构是指相互之间存在着一种或多重关系的数据元素的集合和该集合数据元素之间的关系组成

简单来说, 数据结构就是设计数据以何种方式储进计算机

数据结构的分类: (按照逻辑结构上进行的分类)

线性结构: 数据结构中元素存在着一对一的关系

树结构: 数据结构中存在着一对多的相互关系

图结构: 数据结构中的元素存在着多对多的相互关系

列表便是典型的线性结构, 因为一个元素对应的上一个数和下一个数都是固定的

## 列表

以下谈的所有都是python中的相关知识, 并不适用C

第一个问题: 列表中的元素都是怎么存储的? 顺序存储!

第二个问题: 按下标查找 删除 等等 它们时间复杂度是多少? 

我们先看数组按下表查找: 一定是O(1) 为什么? 内存存住的时候都是有对应标记的,直接一击毙命, 找到对应元素

在32位机上, 一个整数占4个字节, 则对应元素是会有对应的字节占用标志, 方便数组一击毙命找到

那么数组和列表有两点不同: 

1. 数组元素类型一定要相同(否则不同类型占的)
2. 数组长度是固定的

那么列表是怎么做到的呢?

在一个内存条之内, 存的不是元素对应的字节, 而是地址; 因为不同的元素可能在解释期内会被安排存储到不同的位置(一个地址也是占四个字节)

## 栈 STACK

栈stack 是一个数据集合, 可以理解为只能在一段进行插入或者删除操作的列表

栈的特点: 后进先出 (LIFO: last-in  first-out)

栈的概念: 栈定  栈底

栈的基本操作: 进栈(圧栈) push  出栈  pop  取栈定  gettop

怎么支持栈的相关操作呢? 其实只用列表就可以了

li.append    li.pop    li[-1]

## 栈的应用--括号匹配问题

给定一个字符串, 其中考汉小括号中括号大括号, 求该字符串中括号是否匹配

例如 {}[]  正确  {}[(])  错误

设置一个栈, 按顺序每次进入一个, 如果进入的和栈定可以匹配, 则这个元素不进入, 而且栈定会出去

依据: 如果匹配, 则一定会有对称结构, 在对称结构下, 栈不会残留

````python
class Stack:
    def __init__(self):
        self.stack = []
    def push(self,element):
    	self.stack.append(element)
    def pop(self):
    	return self.stack.pop()
    def get_top(self):
        if  len(self.stack) > 0:
            return self.stack[-1]
        else:
        	return False
def brace_match(s):
    stack = Stack()  # 创建空栈, 变量名字是stack
    match = {'(':')','[':']','{':'}'}
    for ch in s:
        if ch in {'(','[]','{'}
            stack.push(ch)  # 进栈
        else:
        	if len(stack) == 0:
        		return False
            elif stack.gettop == match[ch]:
                stack.pop()
            else:
            	return False
    if len(stack) == 0:
        return True
    else:
    	return False
            
````

## 队列 QUEUE

队列是一个数据集合, 仅仅允许在列表的一段进行插入, 另一端进行删除,: 进入插入的一端称为队尾(rear),插入动作称为进队; 进行删除的一段称为队头(front), 删除动作称为出队

那么注意队列的性质: 先进先出(FIFO)

那么如何实现队列呢? 如果是线性的列表, 那么出队的话, 会留下空位, 这十分难操作

因此我们引入环形队列

![image-20230825112409324](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230825112409324.png)

注意最后队满的图: 不是真正的满, 是因为防止无法辨别头尾

注意思想: 并不是让列表中的元素进行移动, 而是头尾对应的下标进行移动; 环形下标对应如何实现呢? 我们可以使用取余数

![image-20230825113108161](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230825113108161.png)

````python
class Queue:
    def __init__ (self,size = 100):
        self.queue = [0 for _ in range(size)]
        self.size = size
        self.rear = 0  # 队尾指针
        self.front = 0  # 队首指针
    def push(self,element):
        if not self.is_filled():
        	self.rear = (self.rear+1)% self.size
        	self.queue[self.rear] = element
        else:
            raise IndexError('Queue is filled')
    def pop(self):
        if not self.is_empty:
            self.front=(self.front+1)%self.size
            return self.queue[self.front]
        else:
            raise IndexError('Queue is empty.')
    def is_empty(self):
        return self.rear == self.front
    def is_filled(self):
        return (self.rear+1)%self.size == self.front
````

当然, 队列的相关代码其实python库里面有

### 队列的内置模块

首先先补充一下什么是双向队列: 就是队首和队尾都可以进出的队列

那么接下来介绍python内置的队列模块

````python
from collections import deque
q = deque() # 如果不传参数就是创建双向空队列
q.append(1)  #  队尾进入队列
q.popleft()  #  队首出队
q.appendleft()  #  队首进入队列
q.pop()  #  队尾出队
q = deque([1,2,3],5)  #  1,2,3会自己进入队伍
#  这个5代表队满时的元素数量,但是不同的地方在于:
#  这个双向队列如果队满后还加元素, 那么队首会自己出列
#  演示读文件并截取要求行数
def tail(n):
    with open('test.txt','r') as f:
        q = deque(f,n)
        return q
for line in tail(5):
    print(line,end='')
#  原理:前面的行数进去后,都会因为队满然后出列,最后只有
#  最后的五行能够留下来
````

### 栈和队列的应用----迷宫问题

![image-20230825140854134](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230825140854134.png)

栈和队列都可以完成这个任务, 而两种方法就对应两种搜索的思路

#### 栈做法----深度优先搜索

![image-20230825141027187](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230825141027187.png)

栈保存路径,  回退一步, 该点就出栈, 一直回到有另一种走法的位置

````python
dirs = [
    lambda x,y:(x+1,y)
    lambda x,y:(x-1,y)
    lambda x,y:(x,y-1)
    lambda x,y:(x,y+1)
]
def maze_path(x1,y1,x2,y2):
    stack = 【】
    stack.append((x1,y1))
    maze【x1】【y1】 = 2
    while(len(stack)>0):
        curNode = stack【-1】 # 当前位置
        if curNode【0】 == x2 and curNode【1】 ==y2:  #  走到终点了
            for p in stack:
                print(p)
            return True
        for dir in dirs:
            nextNode = dir(curNode【0】,curNode【1】)
            if maze[nextNode【0】][nextNode【1】] == 0:
                stack.append(nextNode)
                maze[nextNode【0】][nextNode【1】] = 2  # 把走过的位置标记成2(甚至可以为1), 确保不走回头路
                break  # 尝试这个方向即可!
        else:
            stack.pop()  # 回退
    else: # 正常跳出while : 无路
        print("no path")
        return False

````

#### 队列----广度优先搜索

![image-20230825143659140](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230825143659140.png)

每一次操作, 多个头都同时行动, 可以分出岔路

最大的特点在于: 由于是一次次操作同步进行的, 先到的路径是一定最短的路径

用队列存'头'的坐标, 即当前正在考虑的节点

并且要用额外的列表, 让我们知道一个节点是哪个地方延伸出来的

````python
from collections inport deque
dirs = [
    lambda x,y:(x+1,y)
    lambda x,y:(x-1,y)
    lambda x,y:(x,y-1)
    lambda x,y:(x,y+1)
]
def print_r(path):
    curNode = path[-1] #它仅仅记录了所有节点
    realpath = [] # 注意为什么要再来一个列表
    while curNode[2] != -1:
        realpath.append(curNode[0:2])
        curNode = path[curNode[2]]
    realpath.append(curNode[0:2])# 倒过来的
    realpath.reverse()
    for node in realpath:
        print(node)
def maze_path_deque(x1,y1,x2,y2):
	queue = deque()
    queue.append((x1,y1,-1)) #注意最后的数字是干啥的
    path = []
    while len(queue) > 0:
        curNode = queue.popleft()
        path.append(curNode)
        if curNode[0]==X2 and curNode[1]==y2:
            print_r(path)
        for dir in dirs:
            dir(curNode[0],curNode[1])
            if maze[nextNode[0]][nextNode[1]] == 0:
                queue.append((nextNode[0],nextNode[1],len(path)-1))
# 注意为什么是len(path)-1! nextNode是curNode带进来的! curNode的下标要标记在nextNode信息里面,方便溯源
                maze[nextNode[0]][nextNode[1]]=2
	else:
        print('No path')
        return False              
````

## 链表

### 概念

链表是由一系列节点组成的元素集合, 每个节点包含两部分, 数据域item和指向下一个节点的指针next. 通过节点之间的相互连接, 最终串联形成一个链表

![image-20230826233031930](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230826233031930.png)

````python
class Node:
    def __init__(self,item):
        self.item = item
        self.next = None
a = Node(1)
b = Node(2)
c = Node(3)
a.next = b
b.next = c
print(a.next.item)  #  2
print(a.next.next.item)  #  3
````

![image-20230827001153874](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827001153874.png)

### 创建链表

![image-20230826233546216](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230826233546216.png)

````python
def create_linklist_head(li):  # 最后是倒序
    head = Node(li[0])
    for element in li[1:]:
		node = Node(element)
        node.next = head
    return head
def create_linklist_tail(li):  # 最后是正序
    head = Node(li[0])
    tail = head
    for element in li[1:]
        node = Node(element)
        tail.next = node
        tail = node
    return head
def print_linklist(lk):
    while lk:
        print(lk.item,end = ' ')
        lk = lk.next       
````

### 链表的遍历

![image-20230826235032228](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230826235032228.png)

### 链表节点的插入和删除

它最大的特点就是时间复杂度将不会很大, 如果是列表的话, 是会让其他元素移动的; 但是列表不会

![image-20230826235414645](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230826235414645.png)

![image-20230826235427483](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230826235427483.png)

![image-20230826235509532](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230826235509532.png)

![image-20230826235916261](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230826235916261.png)

### 双链表

之前只能从前往后找, 那么可不可以双向找呢? 

![image-20230827000427534](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827000427534.png)

![image-20230827000816138](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827000816138.png)

![image-20230827000914328](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827000914328.png)

## 哈希表 HASH TABLE

#### 介绍

![image-20230827102432501](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827102432501.png)

![image-20230827102600635](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827102600635.png)

![image-20230827102816175](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827102816175.png)

![image-20230827103029893](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827103029893.png)

![image-20230827103143047](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827103143047.png)

在这个例子中, 如果没有取余数的函数, 那么例如14的东西就不能存进去, 因为只有七个位置

但是如果再来个7呢? 会和14在一个地方; 那么就会发生哈希冲突

![image-20230827103422786](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827103422786.png)

![image-20230827103513504](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827103513504.png)

线性探查其实并不是很好; 二度哈希比较奇怪; 那么现在着重介绍拉链法

![image-20230827103934703](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827103934703.png)

链表的优势就体现出来了, 因为它的元素的插入和删除都是很方便的

常见的哈希函数:

![image-20230827104318825](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827104318825.png)

其中乘法哈希: floor是向下取整, hey%1相当于取小数部分

#### 哈希表的实现:

````python
class LinkList:
    class Node:
		def __init__(self,item = None):
            self.item = item
            self.next = None
    class LinkListIterator:
        def __init__(self,node):
            self.node = node
        def __next__(self):
            if self.node:
				cur_node = self.node
                self.node = cur_node.next
                return cur_node.item
            else:
				raise StopIteration
        def __ iter__(self):
    		return self
        def __init__(self,iterable=None):
            self.head = None
            self.tail = None
            if iterable:
                self.extend(iterable)
        def append(self,obj):
			s = LinkList.Node(obj)
            if not self.head:
                self.haed = s
                self.tail = s
            else:
                self.tail.next = s
                se;f.tail = s
        def extend(self,iterable):
            for obj in iterable:
                self.append(obj)
        def find(self,obj):
            for n in self:
                if n == obj:
                    return True
            else:
                return False
        def __iter__(self):
			return self.LinkListIterator(self.head)
        def __repr__(self):
            return '<<'+', '.join(map(str,self))+'>>'
class HashTable:
    def __init__(self,size = 101):
        self.size = size
        self.T = [None for i in range(self.size)]
    def h(self,k):
        return k % self.size
    def insert(self,k):
        i = self.h(k)
        if self.find(k):
            print('Duplicated Insert.')
        else:
            self.T[i].append(k)            
    def find(self,k):
        i = self.h(k)
        return T[i].find(k)
````

#### 哈希表的应用

![image-20230827110413244](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827110413244.png)

![image-20230827110559800](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827110559800.png)

![image-20230827110958375](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827110958375.png)

![image-20230827111023484](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827111023484.png)

![image-20230827111051652](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230827111051652.png)

## 树

### 树的概念

![image-20230828163312574](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230828163312574.png)

![image-20230828163457764](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230828163457764.png)

## 巩固知识点: 面向对象编程

### 1 理解面向对象

面向对象是一种抽象化的编程思想, 很多编程语言中都有这种思想. 面向对象就是将编程当成是一个事物, 对外界来说, 事物是直接使用的, 不用管他内部的情况. 而编程就是设置事物能够做什么事

举个例子: 要机洗, 只需要找到一台洗衣机, 加入简单操作就可以完成洗衣机的工作, 而不需要关心洗衣机内部发生了什么

所以总结来说, 面向对象编程就是为了简化代码, 就是造出洗衣机方便我们使用

### 2 类和对象

洗衣机是工厂工人制作出来的, 而工厂工人是怎么制作出洗衣机的呢? 工人根据设计师设计的功能图纸去制作洗衣机   所以总结下来就是图纸-洗衣机-洗衣服

在面向对象编程的过程中, 有两个重要组成部分: 类和对象. 两者的关系: 用类去创建(示例)一个对象

#### 2.1 理解类和对象

##### 2.1.1 类

类是一系列具有相同特征和行为的事物的统称, 是一个抽象的概念; 类比如说就是制造洗衣机时要用到的图纸, 也就是说用类来创建对象

特征就是属性, 而行为就是方法; 类是一个抽象的概念, 不是真实存在的事物

##### 2.1.2对象

对象是类创建出来的真实存在的事物, 例如洗衣机; 注意, 在开发中, 现有类, 再有对象

#### 2.2 面向对象实现方法

##### 2.2.1 定义类

python中类分为两种: 经典类 和 新式类

````python
class 类名():
    代码
````

注意类名要满足标识符命名规则, 同时遵守大驼峰命名习惯

##### 2.2.2 创建对象

对象名 = 类名()      对象有名实例

````python
class Washer():
	def wash(self):
		print('能洗衣服')
haier = Washer()
haier.wash()  # 实例方法  对象方法
````

##### 2.2.3 self

self指的是调用该函数的对象:

````python
class Washer():
	def wash(self):
		print('能洗衣服')
        print(self)
haier = Washer()
print(haier)
haier.wash()  
````

试验后会发现, 两次print都会输出相同的内存地址: 即打印对象和打印self得到的内存地址相同, 所以self指的是调用该函数的对象

实例化对象会拥有类的全部方法, 在调用的时候

##### 2.2.4 一个类创建多个对象

一个类是可以创建多个对象的, 这是很容易理解的

### 3 添加和获取对象属性

#### 3.1 类外面添加对象属性

语法: 对象名.属性名 = 值

属性即是特征, 比如: 洗衣机的宽度, 高度, 宽度......

对象属性既可以在类外面添加和获取, 也能在类里面添加和获取

我们为什么要给一个对象添加属性? 

````python
haier1.width = 500
haier1.height = 800  # haier1是对象
````

#### 3.2 类外面获取对象属性

````python
# 语法:  对象名.属性名
print(f'haier1洗衣机的宽度是{haier1.width}')
print(f'haier1洗衣机的高度是{haier1.height}')
````

#### 3.3 类里面获取对象属性

````python
# 语法: self.属性名
class Washer():
    def print_info(self):
		print(f'haier1洗衣机的宽度是{self.width}')
		print(f'haier1洗衣机的高度是{self.height}')
haier1 = Washer()
haier1.width = 500
haier.weight = 800
haier1.print_info()
````

### 4 魔法方法

#### 4.1 __init__()

##### 4.1.1 体验

思考: 洗衣机的宽度和高度是与生俱来的属性, 可不可以在生产过程中就赋予这些特性呢?

理应如此, 那么init()方法的作用: 初始化对象

````python
class Washer():
    def __init__(self):
        self.width = 500
        self.height = 800
    def print_info(self):
        print(f'洗衣机的宽度是{self.width},高度是{self.height}')
haier = Washer()
haier.print_info()
````

##### 4.1.2 带参数的init()

一个类可以创建多个对象, 如何对不同的对象设置不同的初始化属性? 那么我们可以考虑传参数

````python
class Washer():
	def __init__(self,width,height):
        self.width = width
        self.height = height
    def print_info(self):
        print(f'洗衣机的宽度是{self.width},高度是{self.height}')
haier = Washer(10,20)  #  创建一个对象,对象拥有该类的全部方法
haier.print_info()
````

#### 4.2 str()方法

当使用print输出对象的时候, **默认打印对象的内存的地址**, 如果定义了str方法. 那么就会打印这个方法中return的数据

````python
class Washer():
	def __init__(self,width,height):
        self.width = width
        self.height = height
    def __str__(self):
        return '这是海尔洗衣机的说明书'
haier = Washer(10,20)
# 这时候就会打印出这句话
print(haier)
````

#### 4.3 del()方法

当删除对象时, python解释器也会默认调用__del__ ()方法:

````python
class Washer():
	def __init__(self,width,height):
        self.width = width
        self.height = height
    def __del__(self):
        print(f'{self}对象已经被删除')
haier = Washer(10,20)
del haier  # 把对象删除, 那么就会把那句话打印出来
````

### 5 总结

面向对象重要组成部分: 类----创建类    对象  添加对象属性------类外面和类里面   魔法方法

## 贪心算法

![image-20230828164049795](C:\Users\xiong\AppData\Roaming\Typora\typora-user-images\image-20230828164049795.png)

