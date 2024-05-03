import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric: # 定义准确度度量类
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc: # 如果当前实例的acc高于传入实例的acc
            return True
        else:
            return False

    def state_dict(self): # 获取当前实例的状态字典
        _dict = dict()
        _dict['acc'] = self.acc # 将acc添加到字典中 
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    # 使用SVM进行评估的函数
    clf = LinearSVC() # 创建LinearSVC分类器
    clf.fit(train_features, train_labels) # 用训练特征和标签拟合分类器
    pred = clf.predict(test_features) # 使用测试特征进行预测
    # pred 是一个包含测试集每个样本预测结果的 numpy 数组。
    return np.sum(test_labels == pred) * 1. / pred.shape[0]
    # pred.shape[0] 是 pred 数组的第一个维度的大小，即预测结果的样本数量

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # 构建数据集
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank) # 将模型发送到指定的GPU

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0 # 从第0个epoch开始
    best_metrics = Acc_Metric(0.) # 初始化最佳准确度度量为0
    metrics = Acc_Metric(0.) # 初始化当前准确度度量为0

    # resume ckpts 恢复检查点
    if args.resume: # 如果有恢复参数
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger) # 恢复模型和最佳准确度
        best_metrics = Acc_Metric(best_metric) # 更新最佳准确度度量
    elif args.start_ckpts is not None: # 如果有开始检查点参数
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP 如果有恢复参数
    if args.distributed:
        # 同步批量归一化
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model) # 转换为同步批量归一化模型
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # 优化器和调度器
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume: # 如果恢复训练
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad() # 清空梯度
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch) # 设置epoch，用于数据采样
        base_model.train() # 设置模型为训练模式

        epoch_start_time = time.time() # 记录epoch开始时间
        batch_start_time = time.time() # 记录batch开始时间
        batch_time = AverageMeter() # 初始化batch时间度量器
        data_time = AverageMeter() # 初始化数据加载时间度量器
        losses = AverageMeter(['Loss']) # 初始化损失度量器

        num_iter = 0 # 初始化迭代次数

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader) # 获取训练数据批次数量
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx # 计算当前迭代的编号
            
            data_time.update(time.time() - batch_start_time) # 更新数据加载时间
            npoints = config.dataset.train.others.npoints # 获取训练集每批次的点数
            dataset_name = config.dataset.train._base_.NAME # 获取数据集名称
            if dataset_name == 'ShapeNet':
                points = data.cuda() # 将数据发送到GPU
            elif dataset_name == 'ModelNet':
                points = data[0].cuda() # 获取点云数据并发送到GPU
                points = misc.fps(points, npoints) # 使用Farthest Point Sampling采样 
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints # 确保点数匹配
            points = train_transforms(points) # 对点云进行变换
            loss = base_model(points) # 计算损失
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
            # 判断是否已经完成了一次参数更新所需的所有批次的处理。
            # num_iter是当前的训练迭代次数，而config.step_per_update是从配置中获取的步长
                num_iter = 0 # 重置迭代次数
                optimizer.step() # 更新优化器
                base_model.zero_grad() # 清空梯度

            if args.distributed:
                # 将所有节点计算的损失汇总（reduce）成一个总损失。这个过程通常涉及到跨节点通信，
                # 以收集每个节点的损失值并进行累加或平均等操作，确保所有节点都有相同的总损失值。
                loss = dist_utils.reduce_tensor(loss, args)
                # 从损失张量（Tensor）中提取出Python数值。
                # 在PyTorch中，张量（Tensor）是多维的数据数组，而.item()方法将这个张量缩减为一个单一的标量值。
                losses.update([loss.item()*1000])
                # 损失值通常很小，乘以1000可以避免小数点后很多位的显示。
            else:
                losses.update([loss.item()*1000])


            if args.distributed:
                torch.cuda.synchronize()
                # 确保所有GPU的操作在继续之前都已完成，以避免潜在的同步问题。


            if train_writer is not None:
                # 记录当前批次的损失和当前的学习率。
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            # 更新每个批次处理所需的时间，并重置batch_start_time为当前时间，以便计算下一个批次的时间。
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # 每20个批次打印一次训练进度，包括当前epoch、批次数、批次时间、数据加载时间、损失值和当前学习率。
            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        # 在每个epoch结束后，更新学习率调度器。
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch) 
        epoch_end_time = time.time() # 记录每个epoch结束的时间。

        if train_writer is not None: # 如果提供了TensorBoard的写入器，记录整个epoch的平均损失。
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        # 打印每个epoch的训练总结，包括epoch编号、epoch所需时间、平均损失和当前学习率。
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        
        # 保存当前epoch的模型检查点。这个检查点会不断更新
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250: # 如果当前epoch是25的倍数并且大于或等于250，保存一个额外的检查点。
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    # 关闭TensorBoard的写入器。
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    # validate函数用于在每个epoch后评估模型的性能
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    # 收集训练集和测试集的特征和标签，用于支持向量机（SVM）分类器的训练和测试：
    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
    # no_grad()是PyTorch中的一个重要上下文管理器，
    # 用于在代码块内部关闭梯度计算，这有助于减少内存消耗并加快计算速度，因为不需要计算用于反向传播的梯度。
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            # 数据移动到GPU上面
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            # 调用模型以获取输入点云的特征表示，其中noaug=True指示模型在生成特征时不应用任何数据增强。
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            # -1是一个特殊的参数，它告诉PyTorch你希望该维度的大小由其他维度的大小来自动推断。
            target = label.view(-1)
            # 存储特征和标签
            test_features.append(feature.detach())
            test_label.append(target.detach())

        # 一旦收集完所有特征和标签，代码使用torch.cat将它们合并成单个张量：
        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            # dist_utils.gather_tensor函数来跨所有节点聚合特征和标签：
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        # 使用线性支持向量机（LinearSVC）来评估特征的分类性能：
        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    # 函数返回一个Acc_Metric对象，其中包含当前epoch的准确度度量，这可以用于判断是否保存模型的检查点：
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass