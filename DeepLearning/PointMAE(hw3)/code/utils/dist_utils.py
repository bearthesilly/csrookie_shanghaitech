import os

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

# 这些函数通常在分布式训练中使用，以确保不同进程之间的同步和通信。

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    print(f'init distributed in rank {torch.distributed.get_rank()}')


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def reduce_tensor(tensor, args):
    '''
        for acc kind, get the mean in each gpu、
        这个函数用于在所有进程间减少(reduce)张量的值，通常是求和或求平均。
        它首先克隆传入的张量tensor,然后使用torch.distributed.all_reduce来在所有进程间进行reduce操作(默认是求和)。
        最后,它将结果除以world size来计算平均值,并返回这个平均张量。
    '''
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def gather_tensor(tensor, args):
    '''
    这个函数用于收集所有进程生成的张量。它首先创建一个包含与world size相同数量的克隆张量的列表,
    然后使用torch.distributed.all_gather操作来收集所有进程的张量。
    最后,它将收集到的张量沿着第0维(通常是批次维度)拼接起来,并返回这个拼接后的张量.
    '''
    output_tensors = [tensor.clone() for _ in range(args.world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat
