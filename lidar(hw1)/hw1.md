# HW1

![image](https://github.com/bearthesilly/csrookie/blob/main/dl_hw_project/png_hw1/1.png)

这是一个在DBCloud上面利用远程机跑数据集的任务，首先要完成的是和虚拟机的连接：
DBCloud的使用方法，详见：https://d6aihhxw08.feishu.cn/docx/KdgkdJGK0oAEcwxIGuTck9wungb

其中VNC完全按照教程是没有问题的，通过电脑上再下载的VNC Viewer可以连接，提供虚拟桌面

但是SSH连接并不是很顺利。可以通过vscode或者windows cmd的ssh服务连接，但是最终选择了XShell7用于连接和**Xttp7用于传输文件**（这是一个非常好用的功能！）。一开始链接的时候总是SSH连不上，XShell7让我输入密码进行来连接之后，总是报错：``SSH服务器拒绝了密码，请再试一次``一开始以为是虚拟机ssh功能没有开启，但是根据飞书教程排查后发现并不是这样。解决方法是编辑ssh服务器配置文件：

``vi /etc/ssh/sshd_config``

先按s（注意光标处的字母会被吞掉），然后进入编辑模式，修改：
``PermitRootLogin yes`` 

一开始我进行这样的修改之后并没有立马奏效。但是第二天我重新尝试连接的时候就成功了（wtf）

Xttp7连接上了之后，就可以准备配环境了。首先要知道虚拟机的系统，输入以下命令：
``lsb_release -a``, 然后显示我的虚拟机是Ubuntu20.04

如果没有这个指令，需要根据[Linux命令——lsb_release - 克拉默与矩阵 - 博客园 (cnblogs.com)](https://www.cnblogs.com/kelamoyujuzhen/p/9691113.html)去安装这个命令

下载好anaconda 3, cuda, cudnn的压缩包，一并传输到指定文件夹，进行解压。注意，cuda,cudnn,pytorch的版本一定要匹配，cudnn里的文件要覆盖到cuda-version这个文件夹里面。检查nvidia显卡的cuda兼容版本，要输入``nvidia-smi``去查看，cuda版本可以向下兼容，但是pytorch-cuda-cudnn的匹配是最重要的。cuda,cudnn都下载好了之后，conda命令输入：

``conda create --name name python=?``  （注意，=左右不要闲的乱加空格！）

``conda activate name``   , 这样环境就激活了，接下来就是要安装很多库，例如numpy matplotlib  tqdm 

其中最重要的就是install pytorch 了，输入命令一定要查看pytorch官网提供的命令行（以配对cuda等版本）

最后别忘了``conda install cudatoolkit``, 这样一来就可以检查自己的环境是否装对了：(输入python然后enter)

````python
import torch
torch.__version__
torch.cuda.is_available()
````

![image](https://github.com/bearthesilly/csrookie/blob/main/dl_hw_project/png_hw1/2.png)

更多检查指令见上图

然后接下来就是阅读课题组github官网的README，然后根据example的指令指导，验证pretrained model results

过程中遇到的bug or keypoint:

- set PYTHONPATH=C:\path\to\models_or_dataset;%PYTHONPATH%, 将models dataset两个库引入环境变量
- export CUDA_VISIBLE_DEVICES=?,?,...   有的时候其他显卡有人用，这行命令指定哪些GPU跑
- ``conda install pytorch3d``会提示你conda版本过低，没有这个库，这个时候考虑pip + 镜像源

``pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple``

``pip install pytorch3d``   这样就能安装pytorch3d了

（这是如果发生conda下载报错``HTTP 000``的替代方案，如果conda配上镜像源能够实现自然是最好）

- smpl用README中给出的第二个途径进行安装！ 注意``conda install smplx``
- 充分理解cd指令配合 "./ ···"文件路径的表达方式

最后就能得到结果了：

![image](https://github.com/bearthesilly/csrookie/blob/main/dl_hw_project/png_hw1/3.png)

虽然看似卡点不多，但是前前后后一周忙了将近有10个小时。从一开始的docker实例都不敢创建，到终于完成任务，debugging带来的历练是相当大的。Patience is everything（虽然周二下午总是“ssh拒绝了密码”把我搞红温了QAQ）

由于是后来进行的总结，可能上述文档并没有完全记录我遇到的问题或者关键点。而且还有一点感悟：一个报错可能是不同的bug造成的了，这确实是十分annoying的。