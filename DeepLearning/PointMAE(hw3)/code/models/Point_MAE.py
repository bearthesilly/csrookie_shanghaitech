import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        '''
         nn.Sequential是一个容器,用于包装一系列神经网络层或模块,使其能够按顺序执行。
        这个容器非常适合用来构建一个简单的、按顺序执行的神经网络模型，其中每一层都会在前一层的输出上进行操作。
        以简化模型的构建过程,特别是当模型由多个层顺序堆叠而成时。你只需要将这些层或模块添加到nn.Sequential中,
        PyTorch就会自动按照它们添加的顺序来构建模型。
        '''
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), # 将输入的3维特征（点云中的x, y, z坐标）转换成128维的特征，使用1维卷积
            nn.BatchNorm1d(128), # 批量归一化层，用于规范化上一层的输出
            nn.ReLU(inplace=True), # 参数inplace=True意味着ReLU的计算将在输入数据的内存位置上直接进行
            nn.Conv1d(128, 256, 1) # 将128维的特征进一步转换成256维
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), # 512 = 256(first_conv得到) +(concat) 256(maxpooling)
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1) # 将512维的特征转换成encoder_channel维，这个维度是模型配置中指定的
        )
        '''
        nn.Conv1d(in_channels, out_channels, kernel_size):
        输入数据的通道数;输出数据的通道数;卷积核（或滤波器）的大小
        '''

    def forward(self, point_groups):
        # 前向传播函数，处理输入的点云数据并生成特征表示
        '''
            point_groups : B G N 3
            其中B是批次大小,G是每个批次中的组数,N是每个组中的点数,3是因为每个点的坐标是三维的
            为什么有"组"?下面的class Group就是在分组!
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape # batch_size, 一个batch里面点组数,一组里面点的数量,一个点的三维信息
        point_groups = point_groups.reshape(bs * g, n, 3) 
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        # 将维度从(bs * g, n, 3)转换为(bs * g, 3, n)，这是因为nn.Conv1d期望第一个维度是通道数;匹配输入格式
        # nn.Conv1d 期望输入张量遵循一定的格式，这个格式通常表示为 (batch_size, channels, length)
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 maxpooling
        # torch.max返回的是元组,只取第一个(要值而不是引索);torch.Size([batch_size, channels, 1])
        # "1"的存在是因为keepdim = True
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        # feature_global.expand(-1, -1, n) 将 feature_global 张量沿 length 维度扩展 n 次，而不改变其他维度。-1 表示该维度保持原有大小。
        # dim = 1代表沿着拼接的维度
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN 如何进行分组
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group # 存储要采样的中心点的数量，即组数
        self.group_size = group_size # 每个组内的点数
        self.knn = KNN(k=self.group_size, transpose_mode=True) # 创建一个KNN类的实例，用于执行k最近邻搜索

    def forward(self, xyz):
        '''
            input: B N 3
            输入xyz的维度是B N 3,其中B是批次大小,N是点云中的点数,3表示每个点的三维坐标
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3  G是self.num_group
        # center就是选中的num_group个中心点 
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M 只关心索引，不关心实际的距离
        # idx是   tensor.Size([batch_size, num_group, group_size])
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        '''
        KNN搜索返回的索引是相对于每个中心点的局部索引.
        torch.arange(0, batch_size, device=xyz.device):创建一个从0到batch_size(不包括batch_size)的整数序列。
        batch_size是输入点云数据的批次大小。device=xyz.device确保这个序列在与点云数据相同的设备上
        view(-1, 1, 1)：将上述创建的一维整数序列重塑为一个三维张量，其形状为(batch_size, 1, 1)。
        这里的-1是让PyTorch自动计算该维度的大小,以便保持元素总数不变。
        * num_points:将每个元素在最后一个维度上乘以num_points(每一批次点的数量)
        最终,idx_base是一个张量,其作用是为每个批次中的每个点提供一个全局的偏移量索引。
        这个偏移量索引随后用于将局部邻域索引(idx)转换为原始点云中的全局索引。
        原来idx里面全都是相对于一个batch下的引索,从0到num_points-1;xyz里面引索是全局的
        '''
        idx = idx + idx_base # 得到点云中的全局索引
        idx = idx.view(-1) #  将 idx 张量重塑为一个一维张量 
        '''
        为什么需要将索引展平为一维？在处理点云数据时，我们通常需要根据这些索引从原始点云中提取特定的点。
        通过将索引展平为一维，我们可以更方便地使用这些索引来索引原始点云张量
        '''
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        '''
        .view()改变张量的形状而不改变其数据类型
        [idx, :]::表示选取每个索引行中的所有列（即每个点的所有三维坐标）
        neighborhood张量包含了根据idx索引从xyz中提取的邻域点。其形状是(B * G * M, 3)，其中每个点的三维坐标是连续排列的。
        '''
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2) # 将邻域点相对于中心点进行归一化，即从每个邻域点中减去对应的中心点坐标
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # GELU（Gaussian Error Linear Unit）
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 如果是None,就用后面的值
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() # 激活层的类型，默认为nn.GELU（高斯误差线性单元）
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) # 防止过拟合

    def forward(self, x):
        '''
        在forward方法中,输入x首先通过第一个全连接层fc1,然后通过激活函数act,接着通过Dropout层drop进行正则化。
        这一过程重复一次,首先通过第二个全连接层fc2,再次应用激活函数和Dropout。最终,处理后的张量x被返回
        多层感知机是Transformer模型中的一个重要组成部分,它在自注意力机制之后应用,用于进一步处理和提取特征
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # reshape和permute操作将Q、K、V的表示重排并分割成多个头，以实现多头注意力机制。
        '''
        permutate的含义:
        2:原来表示Q、K、V的维度,现在放在第一位,这样每个头的Q、K、V可以连续存储。
        0:原来的批次大小维度，现在放在第二位。
        3:头数维度，现在放在第三位。
        1:原来的序列长度维度，现在放在第四位。
        4:每个头的特征维度，现在放在最后一位。
        '''
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # @是阿达玛积
        # 计算查询和键的点积，然后乘以缩放因子。
        attn = attn.softmax(dim=-1) # 对注意力分数进行softmax归一化，得到注意力权重
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 使用注意力权重和值（V）计算加权和 
        x = self.proj(x) # 使用注意力权重和值（V）计算加权和
        x = self.proj_drop(x) # 在变换后的输出上应用Dropout
        return x
    '''
    self.num_heads:设置注意力机制的头数。
    head_dim:计算每个头的特征维度,这是通过将输入维度dim除以头数num_heads得到的。
    self.scale:设置注意力分数的缩放因子,通常使用head_dim的负0.5次方，这是为了在计算注意力分数时进行缩放，以保持梯度的稳定性。
    self.qkv:定义一个线性层,用于生成查询(Query, Q)、键(Key, K)和值(Value, V)的表示。它将输入特征维度扩展为dim * 3,因为每个输入需要生成Q、K、V三个输出。
    self.attn_drop:定义一个Dropout层,用于在注意力分数上应用Dropout,以进行正则化。
    self.proj:定义一个线性层，用于在注意力机制的输出上进行变换。
    self.proj_drop:定义另一个Dropout层,用于在变换后的输出上应用Dropout。
    '''


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 然后输出通过第二层归一化self.norm2,进入MLPself.mlp,再通过DropPath正则化。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    '''
    首先输入x通过第一层归一化self.norm1,然后通过自注意力机制self.attn,最后通过DropPath正则化。
    接着将DropPath的输出与原始输入x相加,进行残差连接(residual connection),这有助于避免在深层网络中出现梯度消失的问题。
    然后输出通过第二层归一化self.norm2,进入MLPself.mlp,再通过DropPath正则化。
    最后MLP的输出再次与原始输入相加,进行连接。
    '''

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])
    '''
    embed_dim:输入特征的维度。
    depth:Transformer编码器块的数量。
    num_heads:每个自注意力机制中的头数。
    mlp_ratio:多层感知机(MLP)隐藏层维度与输入维度的比例。
    qkv_bias:是否在线性层中添加偏置。
    qk_scale:注意力机制中的缩放因子。
    drop_rate:MLP中Dropout的比率。
    attn_drop_rate:自注意力机制中Dropout的比率。
    drop_path_rate:DropPath正则化的比率,可以是一个列表或单个值
    '''
    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x
    '''
    使用一个for循环遍历self.blocks中的每个Block实例。
    在每次迭代中,将当前块应用于输入x,同时将x与位置编码pos相加,以将位置信息融入模型的输入中。这种位置编码通常是必要的,
    因为Transformer模型本身不具备捕捉序列中位置关系的能力。每个块的输出会作为下一个块的输入。
    最后,循环结束后的x作为整个编码器的输出返回
    '''


class TransformerDecoder(nn.Module):
    # Transformer解码器逐步处理输入数据的特征表示，并将解码器的输出用于生成或预测任务
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) # 一个层归一化层，用于规范化解码器的输出
        self.head = nn.Identity()

        self.apply(self._init_weights) # 使用_init_weights方法对模型的权重进行初始化
    '''
    embed_dim:输入特征的维度。
    depth:Transformer解码器块的数量。
    num_heads:每个自注意力机制中的头数。
    mlp_ratio:多层感知机(MLP)隐藏层维度与输入维度的比例。
    qkv_bias:是否在线性层中添加偏置。
    qk_scale:注意力机制中的缩放因子。
    drop_rate:MLP中Dropout的比率。
    attn_drop_rate:自注意力机制中Dropout的比率。
    drop_path_rate:DropPath正则化的比率,可以是一个列表或单个值。
    norm_layer:归一化层的类型,默认为nn.LayerNorm。
    '''
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) # 对于线性层（nn.Linear），使用Xavier均匀分布初始化权重，并初始化偏置为0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        # 只取x的最后一部分（由return_token_num指定的元素数量），经过层归一化和恒等映射处理后作为输出
        return x
    '''
    使用一个for循环遍历self.blocks中的每个Block实例。
    在每次迭代中,将当前块应用于输入x,同时将x与位置编码pos相加,以融入位置信息。
    循环结束后,通过层归一化self.norm和恒等映射self.head处理x的最后return_token_num个元素,
    通常这些元素代表解码器的输出,如预测的像素或标记。
    '''


# Pretrain model
class MaskTransformer(nn.Module):
    # 实现了一个带有掩码操作的Transformer编码器，这在自监督学习中常用于数据增强和特征提取
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        # 生成基于点云中心的块状掩码
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        # 生成基于随机选择的掩码
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G  这是一个布尔类型的张量

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # 从 group_input_tokens 中选取那些未被掩码的点，即保留 bool_masked_pos 为 False 的位置对应的点。
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos
    '''
    根据掩码类型生成掩码。
    使用Encoder对邻域点云进行编码,得到输入特征。
    从编码的特征中选择未被掩码的部分,并将其重塑为适合Transformer处理的形状。
    对掩码的中心点应用位置嵌入。
    将未掩码的特征和位置信息传递给TransformerEncoder进行处理。
    通过层归一化对输出进行归一化。
    返回处理后的特征x_vis和掩码信息bool_masked_pos
    '''


@MODELS.register_module()
class Point_MAE(nn.Module):
    '''
    super().__init__():调用基类nn.Module的构造函数。
    self.config:存储模型的配置信息。
    self.MAE_encoder:创建一个MaskTransformer实例,用于编码器部分。
    self.group_size和self.num_group:定义点云分组的大小和数量。
    self.mask_token:定义一个掩码标记，用于在解码器中表示掩码位置。
    self.decoder_pos_embed:定义一个位置嵌入网络，用于将点的位置信息编码成特征。
    self.MAE_decoder:创建一个TransformerDecoder实例,用于解码器部分。
    self.group_divider:创建一个Group实例,用于将点云分割成多个局部组。
    self.increase_dim:定义一个网络，用于将解码器的输出增加到所需的维度。
    self.build_loss_func(self.loss):根据配置中的损失函数类型,构建损失函数。
    '''
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, pts, vis = False, **kwargs):
        '''
        使用self.group_divider将点云分割成多个局部组。
        使用self.MAE_encoder对点云进行编码,得到可见部分的特征和掩码。
        对中心点应用位置嵌入，生成可见点和掩码点的位置嵌入。
        将可见特征、掩码标记和位置嵌入组合m输入到self.MAE_decoder进行解码。
        使用self.increase_dim将解码器的输出转换为点云的预测。
        计算重建的点云和原始点云之间的损失。
        如果vis标志为真m生成可视化所需的数据并返回:否则m只返回损失。
        '''
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B,_,C = x_vis.shape # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B*M,-1,3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1
   

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        '''
        super().__init__():调用基类nn.Module的构造函数。
        self.config:存储模型的配置信息。
        self.group_divider:创建一个Group实例,用于将点云分割成多个局部组。
        self.encoder:创建一个Encoder实例,用于将点云数据编码成特征。
        self.cls_token和self.cls_pos:分别定义分类标记和位置标记,这些标记将用于Transformer的自注意力机制。
        self.pos_embed:定义一个位置嵌入网络，用于将点的位置信息编码成特征。
        self.blocks:创建一个TransformerEncoder实例,用于处理特征。
        self.norm:添加一个层归一化层。
        self.cls_head_finetune:定义一个分类头网络，用于最终的分类预测。
        self.build_loss_func():构建损失函数。
        trunc_normal_(self.cls_token, std=.02)和trunc_normal_(self.cls_pos, std=.02)：使用截断正态分布初始化分类标记和位置标记。
        '''
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss() # 初始化交叉熵损失函数

    def get_loss_acc(self, ret, gt): # 计算模型输出ret和目标gt之间的损失和准确度
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ''''
        从给定的检查点路径bert_ckpt_path加载模型权重。
        如果检查点存在,将权重加载到模型中:否则,打印从零开始训练的消息,并应用_init_weights方法进行权重初始化。
        '''
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m): # 使用不同的策略初始化模型中的线性层、层归一化层和一维卷积层
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
    '''
    forward方法是模型的前向传播函数,它接收点云数据pts,并返回分类结果。
    使用self.group_divider将点云分割成多个局部组。
    使用self.encoder对点云进行编码,得到输入特征。
    准备分类标记和位置标记，并将它们与输入特征连接。
    对连接后的特征应用位置嵌入。
    将特征和位置信息传递给TransformerEncoder进行处理。
    通过层归一化对输出进行归一化。
    将归一化后的输出的第一个标记（分类标记）和剩余标记的最大值连接，形成分类特征。
    使用分类头网络对分类特征进行处理，得到最终的分类结果
    '''
