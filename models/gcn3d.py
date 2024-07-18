"""
@Author: Zhi-Hao Lin
@Contact: r08942062@ntu.edu.tw
@Time: 2020/03/06
@Document: Basic operation/blocks of 3D-GCN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices ** 2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """

    bs, v, n = index.size()

    # ss = time.time()
    if bs == 1:
        # id_0 = torch.arange(bs).view(-1, 1,1)
        tensor_indexed = tensor[torch.Tensor([[0]]).long(), index[0]].unsqueeze(dim=0)
    else:
        id_0 = torch.arange(bs).view(-1, 1, 1).long()
        tensor_indexed = tensor[id_0, index]
    # ee = time.time()
    # print('tensor_indexed time: ', str(ee - ss))
    return tensor_indexed


def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    # ss = time.time()
    neighbors = indexing_neighbor(vertices, neighbor_index)  # (bs, v, n, 3)

    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    return neighbor_direction_norm.float()


class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""

    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.SiLU(inplace=True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_num)",
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        # ss = time.time()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        # R = get_rotation(0,0,0)
        # R = torch.from_numpy(R).cuda()
        # R = R.unsqueeze(0).repeat(bs,1,1).float() ## bs 3,3
        # vertices2 = torch.bmm(R,vertices.transpose(1,2)).transpose(2,1)
        # neighbor_direction_norm2 = get_neighbor_direction_norm(vertices2, neighbor_index)

        support_direction_norm = F.normalize(self.directions, dim=0)  # (3, s * k)

        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim=2)[0]  # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim=2)  # (bs, vertice_num, kernel_num)
        return feature


class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.SiLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim=0)
        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias  # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:]  # (bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support,
                                            neighbor_index)  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support  # (bs, vertice_num, out_channel)
        return feature_fuse


class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map,
                                             neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]  # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool

class GCN3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, neighbor_num=10, support_num=7, **kwargs):
        super(GCN3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neighbor_num = neighbor_num
        self.support_num = support_num
        
        self.conv_0 = Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = Conv_layer(in_channel=128, out_channel=128, support_num=self.support_num)
        self.pool_1 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = Conv_layer(in_channel=128, out_channel=256, support_num=self.support_num)
        self.conv_3 = Conv_layer(in_channel=256, out_channel=256, support_num=self.support_num)
        self.pool_2 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = Conv_layer(in_channel=256, out_channel=512, support_num=self.support_num)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        
        bs, _, _ = x.shape
        
        neighbor_index = get_neighbor_index(x, self.neighbor_num)
        
        fm_0 = F.silu(self.conv_0(neighbor_index, x), inplace=True) # B 1024 128
        
        fm_1 = F.silu(self.bn1(self.conv_1(neighbor_index, x, fm_0).transpose(1, 2)).transpose(1, 2), inplace=True) # B 1024 128
        
        v_pool_1, fm_pool_1 = self.pool_1(x, fm_1) # B 256 3 ; B 256 128
        
        neighbor_index = get_neighbor_index(v_pool_1, min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.silu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),inplace=True) # B 256 256
        
        fm_3 = F.silu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),inplace=True) # B 256 256
        
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3) # B 64 3; B 64 256
        
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = get_neighbor_index(v_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2) # B 64 512
        
        # f_global = fm_4.max(1)[0]  # (bs, f) B 512

        nearest_pool_1 = get_nearest_index(x, v_pool_1)
        nearest_pool_2 = get_nearest_index(x, v_pool_2)
        fm_2 = indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=-1) # B 1024 1280
        
        return feat

    
def test():
    import time
    bs = 8
    v = 1024
    dim = 3
    n = 10
    neighbor_num = n
    vertice_num = v
    vertices = torch.randn(bs, v, dim)
    neighbor_index = get_neighbor_index(vertices, n)

    s = 7
    conv_0 = Conv_surface(kernel_num=128, support_num=s)
    conv_1 = Conv_layer(in_channel=128, out_channel=128, support_num=s)
    pool_1 = Pool_layer(pooling_rate=4, neighbor_num=4)
    conv_2 = Conv_layer(in_channel=128, out_channel=256, support_num=s)
    conv_3 = Conv_layer(in_channel=256, out_channel=256, support_num=s)
    pool_2 = Pool_layer(pooling_rate=4, neighbor_num=4)
    conv_4 = Conv_layer(in_channel=256, out_channel=512, support_num=s)
    
    bn1 = nn.BatchNorm1d(128)
    bn2 = nn.BatchNorm1d(256)
    bn3 = nn.BatchNorm1d(256)
    
    recon_num=3
    recon_head = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.SiLU(inplace=True),
                nn.Conv1d(128, 64, 1),
                nn.SiLU(inplace=True),
                nn.Conv1d(64, recon_num, 1),
            )
    
    dim_fuse = sum([128, 128, 256, 256, 512])
    conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
        )

    print("Input size: {}".format(vertices.size()))
    start = time.time()
    fm_0 = F.relu(conv_0(neighbor_index, vertices), inplace=True) # B 1024 128
    print("[1] fm_0 Out shape: {}".format(fm_0.size()))
    
    fm_1 = F.relu(bn1(conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2), inplace=True) # B 1024 128
    print("[2] fm_1 Out shape: {}".format(fm_1.size()))
    
    v_pool_1, fm_pool_1 = pool_1(vertices, fm_1) # B 256 3 ; B 256 128
    print("[3] v shape: {}, f shape: {}".format(v_pool_1.size(), fm_pool_1.size()))
    
    neighbor_index = get_neighbor_index(v_pool_1, min(neighbor_num, v_pool_1.shape[1] // 8))
    fm_2 = F.relu(bn2(conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),inplace=True) # B 256 256
    print("[4] fm_2 Out shape: {}".format(fm_2.size()))
    
    fm_3 = F.relu(bn3(conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),inplace=True) # B 256 256
    print("[5] fm_3 Out shape: {}".format(fm_3.size()))
    
    v_pool_2, fm_pool_2 = pool_2(v_pool_1, fm_3) # B 64 3; B 64 256
    print("[6] v shape: {}, f shape: {}".format(v_pool_2.size(), fm_pool_2.size()))
    
    # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
    neighbor_index = get_neighbor_index(v_pool_2, min(neighbor_num, v_pool_2.shape[1] // 8))
    fm_4 = conv_4(neighbor_index, v_pool_2, fm_pool_2) # B 64 512
    print("[7] fm_4 Out shape: {}".format(fm_4.size()))
    
    f_global = fm_4.max(1)[0]  # (bs, f) B 512
    print("[8] f_global Out shape: {}".format(f_global.size()))
    

    nearest_pool_1 = get_nearest_index(vertices, v_pool_1)
    nearest_pool_2 = get_nearest_index(vertices, v_pool_2)
    fm_2 = indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
    fm_3 = indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
    fm_4 = indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)

    feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=-1) # B 1024 1280
    print("[9] feat Out shape: {}".format(feat.size()))
    
    feat_face_re = f_global.view(bs, 1, f_global.shape[1]).repeat(1, feat.shape[1], 1).permute(0, 2, 1) # B 512 1024
    print("[10] feat_face_re Out shape: {}".format(feat_face_re.size()))
    
    conv1d_input = feat.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num) # B 1280 1024
    conv1d_out = conv1d_block(conv1d_input) # B 256 1024
    print("[11] conv1d_out Out shape: {}".format(conv1d_out.size()))

    recon = recon_head(conv1d_out) # B 3 1024
    print("[12] recon Out shape: {}".format(recon.size()))
    
    recon = recon.permute(0, 2, 1) # B 1024 3 
    print("[13] recon Out shape: {}".format(recon.size()))
    
    # f1 = conv_1(neighbor_index, vertices)
    # # print("\n[1] Time: {}".format(time.time() - start))
    # print("[1] Out shape: {}".format(f1.size()))
    # start = time.time()
    # f2 = conv_2(neighbor_index, vertices, f1)
    # # print("\n[2] Time: {}".format(time.time() - start))
    # print("[2] Out shape: {}".format(f2.size()))
    # start = time.time()
    # v_pool, f_pool = pool(vertices, f2)
    # # print("\n[3] Time: {}".format(time.time() - start))
    # print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    # test()
    
    bs = 8
    v = 1024
    dim = 3
    n = 10
    neighbor_num = n
    vertice_num = v
    x = torch.randn(bs, v, dim)
    gcn3d = GCN3D()
    output = gcn3d(x)
    print(output['recon'].shape)