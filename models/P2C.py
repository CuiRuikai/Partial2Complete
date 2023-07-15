import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.build import MODELS
from models.transformer import Group
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from timm.models.layers import trunc_normal_
from utils.logger import *


class Encoder(nn.Module):
    def __init__(self, feat_dim):
        """
        PCN based encoder
        """
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feat_dim, 1)
        )

    def forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024
        return feature_global


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )

    def forward(self, z):
        bs = z.size(0)

        pcd = self.mlp1(z).reshape(bs, -1, 3)  #  B M C(3)

        return pcd


class ManifoldnessConstraint(nn.Module):
    """
    The Normal Consistency Constraint
    """
    def __init__(self, support=8, neighborhood_size=32):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        self.support = support
        self.neighborhood_size = neighborhood_size

    def forward(self, xyz):

        normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)

        idx = pointops.knn(xyz, xyz, self.support)[0]
        neighborhood = pointops.index_points(normals, idx)

        cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)
        penalty = 1 - cos_similarity
        penalty = penalty.std(-1)
        penalty = penalty.mean(-1)
        return penalty


@MODELS.register_module()
class P2C(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # define parameters
        self.config = config
        self.num_group = config.num_group
        self.group_size = config.group_size
        self.mask_ratio = config.mask_ratio
        self.feat_dim = config.feat_dim
        self.n_points = config.n_points
        self.nbr_ratio = config.nbr_ratio

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(self.feat_dim)
        self.generator = Decoder(latent_dim=self.feat_dim, num_output=self.n_points)

        # init weights
        self.apply(self._init_weights)
        # init loss
        self._get_lossfnc_and_weights(config)

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

    def _get_lossfnc_and_weights(self, config):
        # define loss functions
        self.shape_criterion = ChamferDistanceL1()
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        self.manifold_constraint = ManifoldnessConstraint(support=config.support, neighborhood_size=config.neighborhood_size)
        self.shape_matching_weight = config.shape_matching_weight
        self.shape_recon_weight = config.shape_recon_weight
        self.latent_weight = config.latent_weight
        self.manifold_weight = config.manifold_weight

    def _group_points(self, nbrs, center, B, G):
        nbr_groups = []
        center_groups = []
        perm = torch.randperm(G)
        acc = 0
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts):
        # group points
        nbrs , center = self.group_divider(pts)  # neighborhood, center
        B, G, _ = center.shape
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        feat  = self.encoder(rebuild_points.view(B, -1, 3))

        # complete shape generation
        pred = self.generator(feat).contiguous()

        # shape reconstruction loss
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        # shape completion loss
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        # latent reconstruction loss
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        feat_recon = self.encoder(nbrs_pred.view(B, -1, 3).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)
        # normal consistency constraint
        manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + manifold_penalty

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, manifold_penalty


    def forward(self, partial, n_points=None, record=False):
        # group points
        B, _, _ = partial.shape
        feat = self.encoder(partial)
        pred = self.generator(feat).contiguous()
        return pred
