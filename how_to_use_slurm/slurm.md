# 如何使用slurm（自学）

主要来源：B站BV1Uu411n7jT； 学校手册：http://10.15.89.177:8889/job/index.html

## 调度系统是什么

<img src="png/1.png" alt="image" style="zoom: 50%;" />

sbatch提交作业，然后调度器按照一定规则，会分配至相对应的计算节点

## 作业相关

 **非交互式**作业提交方式：``sbatch xxxx.sh``

![image](png/2.png)

上面的参数中： ``-p normal ``代表分区分到normal区； -n代表GPU核心数量； -mem代表内存数量； -N是节点数量

注意：这里面的井号不是注释的意思！务必带上！

取消作业： ``scancel job_id`

查看集群中节点状态和资源信息： ``sinfo``

![image](png/3.png)

alloc就是正在申请，idle就是空闲，down就是故障

``squeu``查看当前在集群中排队和运行的作业情况

``sacct``查看作业的执行和计费信息









