from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

def main():
    # args，下面语句返回的是命名空间对象，包含了所有解析后的参数
    args = parser.get_args()
    # CUDA，判断CUDA是否可用
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu: # 如果可以用CUDA
        torch.backends.cudnn.benchmark = True # 设置CUDNN基准测试为True，以提高性能
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none': # 如果启动器为'none'
        args.distributed = False # 不使用分布式
    else:
        args.distributed = True # 使用分布式
        dist_utils.init_dist(args.launcher) # 初始化分布式环境
        # 在分布式训练模式下重新设置gpu_ids
        _, world_size = dist_utils.get_dist_info() # 获取分布式信息
        args.world_size = world_size # 设置世界大小（节点数）
    # 日志相关
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime()) # 获取当前时间戳
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log') # 设置日志文件路径
    logger = get_root_logger(log_file=log_file, name=args.log_name) # 获取日志记录器
    # 定义TensorBoard的写入器
    if not args.test:
        if args.local_rank == 0: # 如果是主节点
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
            # 创建训练与验证日志写入器
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    # batch size 根据是否分布式以及总批量大小调整训练和验证的批量大小
    if args.distributed:
        assert config.total_bs % world_size == 0 # 确保总批量大小可以被节点数整除
        config.dataset.train.others.bs = config.total_bs // world_size # 计算每个节点的批量大小
        # config.dataset：用来存储数据集相关的配置，可能包括训练集、验证集、测试集等的相关配置信息
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size 
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs 
    # log 将命令行参数和配置保存到日志文件
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}') # 日志记录是否使用分布式训练
    # set random seeds 设置随机种子以确保实验的可重复性
    # 随机性通常会涉及到参数初始化、数据集划分、样本顺序等方面。设置随机种子可以在相同的实验条件下获得相同的随机结果
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() # 确保本地rank与分布式rank相匹配

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold
    '''
    这段代码是在检查是否传入了名为 shot 的参数，并根据参数值设置数据集的 shot、way 和 fold 参数。这种设置通常用于少样本学习(few-shot learning)等场景，
    其中 shot 表示每个类别的样本数, way 表示类别数,fold 表示折数（用于交叉验证等）.如果传入了 shot 参数，则将其值分别设置给训练集和验证集的 shot、way 
    和 fold 参数,以便在训练和验证时使用.
    ''' 
    # run
    if args.test: # 如果是测试模式，则调用测试网络
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model: # 如果是微调模型或从头开始训练
            finetune(args, config, train_writer, val_writer) # 调用微调网络
        else:
            pretrain(args, config, train_writer, val_writer) # 调用预训练网络


if __name__ == '__main__':
    main()
