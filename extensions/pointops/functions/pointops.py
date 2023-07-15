import torch
from torch.autograd import Function

import pointops_cuda


def knn(x, src, k, transpose=False):
    if transpose:
        x = x.transpose(1, 2).contiguous()
        src = src.transpose(1, 2).contiguous()
    b, n, _ = x.shape
    m = src.shape[1]
    x = x.view(-1, 3)
    src = src.view(-1, 3)
    x_offset = torch.full((b,), n, dtype=torch.long, device=x.device)
    src_offset = torch.full((b,), m, dtype=torch.long, device=x.device)
    x_offset = torch.cumsum(x_offset, dim=0).int()
    src_offset = torch.cumsum(src_offset, dim=0).int()
    idx, dists = knnquery(k, src, x, src_offset, x_offset)
    idx = idx.view(b, n, k) - (src_offset - m)[:, None, None]
    return idx.long(), dists.view(b, n, k)


def fps(x, k):
    b, n, _ = x.shape
    x = x.view(-1, 3)
    offset = torch.full((b,), n, dtype=torch.long, device=x.device)
    new_offset = torch.full((b,), k, dtype=torch.long, device=x.device)
    offset = torch.cumsum(offset, dim=0).int()
    new_offset = torch.cumsum(new_offset, dim=0).int()
    idx = furthestsampling(x, offset, new_offset).long()
    return x[idx].view(b, k, 3)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx

furthestsampling = FurthestSampling.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)

knnquery = KNNQuery.apply
