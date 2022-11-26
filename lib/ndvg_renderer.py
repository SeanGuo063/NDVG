import os
import time
import math
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils

'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 nearest=False, pre_act_density=False, in_act_density=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 posbase_pe=5, viewbase_pe=4, use_appcode=False,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        print("----------Init Radiance Grid-----------")
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        print("dvgo: xyz_min, xyz_max: ", self.xyz_min, ", ", self.xyz_max)
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest
        self.pre_act_density = pre_act_density
        self.in_act_density = in_act_density
        if self.pre_act_density:
            print('dvgo: using pre_act_density may results in worse quality !!')
        if self.in_act_density:
            print('dvgo: using in_act_density may results in worse quality !!')

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'posbase_pe': posbase_pe, 'viewbase_pe': viewbase_pe, 'use_appcode': use_appcode,
        }
        self.rgbnet_full_implicit = rgbnet_full_implicit
        self.use_appcode = use_appcode
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*posbase_pe*2) + (3+3*viewbase_pe*2)
            if self.use_appcode:
                dim0 += 1 + viewbase_pe * 2
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dvgo: feature voxel grid', self.k0.shape)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            self.mask_cache = MaskCache(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self._set_nonempty_mask()
        else:
            self.mask_cache = None
            self.nonempty_mask = None
        print("--------------- Finish ----------------")

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_color_thres': self.fast_color_thres,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        self.density[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation(self):
        tv = total_variation(self.activate_density(self.density, 1), self.nonempty_mask)
        return tv

    def k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox

    def forward(self, rays_pts, mask_outbbox, interval, viewdirs, occlusion_mask=None, app_code=None, **render_kwargs):
        '''
            give alpha and rgb according to given positions
        '''

        # update mask for query points in known free space
        if self.mask_cache is not None:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        # query for alpha
        alpha = torch.zeros_like(rays_pts[...,0])
        if self.pre_act_density:
            # pre-activation
            alpha[~mask_outbbox] = self.grid_sampler(
                    rays_pts[~mask_outbbox], self.activate_density(self.density, interval))
        elif self.in_act_density:
            # in-activation
            density = self.grid_sampler(rays_pts[~mask_outbbox], F.softplus(self.density + self.act_shift))
            alpha[~mask_outbbox] = 1 - torch.exp(-density * interval)
        else:
            # post-activation
            density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
            alpha[~mask_outbbox] = self.activate_density(density, interval)
        if occlusion_mask is not None:
            alpha = alpha * occlusion_mask.squeeze()
        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        mask = (weights > self.fast_color_thres)
        k0 = torch.zeros(*weights.shape, self.k0_dim).to(weights)
        if not self.rgbnet_full_implicit:
            k0[mask] = self.grid_sampler(rays_pts[mask], self.k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[..., 3:]
                k0_diffuse = k0[..., :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
            rays_xyz = rays_pts[mask]
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)
            timestep_emb = (app_code.unsqueeze(-1) * self.viewfreq).flatten(-2)
            timestep_emb = torch.cat([app_code, timestep_emb.sin(), timestep_emb.cos()], -1)
            if self.use_appcode:
                rgb_feat = torch.cat([
                    k0_view[mask],
                    xyz_emb,
                    # TODO: use `rearrange' to make it readable
                    viewdirs_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)],
                    timestep_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)],
                ], -1)
            else:
                rgb_feat = torch.cat([
                    k0_view[mask],
                    xyz_emb,
                    # TODO: use `rearrange' to make it readable
                    viewdirs_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)],
                ], -1)
            rgb_logit = torch.zeros(*weights.shape, 3).to(weights)
            rgb_logit[mask] = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb_logit[mask] = rgb_logit[mask] + k0_diffuse
                rgb = torch.sigmoid(rgb_logit)
        if occlusion_mask is not None:
            rgb = rgb * occlusion_mask
        return  alpha, alphainv_cum, rgb, weights, mask


class DeformVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 deform_num_voxels=0, deform_num_voxels_base=0,
                 nearest=False, pre_act_density=False, in_act_density=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_deform_thres=0,
                 deformnet_dim=0, deformnet_full_implicit=False,
                 deformnet_depth=3, deformnet_width=128, deformnet_output=3,
                 posbase_pe=5, timebase_pe=5,
                 train_times=None,
                 **kwargs):
        super(DeformVoxGO, self).__init__()
        print("---------Init Deformation Grid---------")
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        print("dvgo: xyz_min, xyz_max: ", self.xyz_min, ", ", self.xyz_max)
        self.fast_deform_thres = fast_deform_thres
        self.nearest = nearest
        self.pre_act_density = pre_act_density
        self.in_act_density = in_act_density
        if self.pre_act_density:
            print('dvgo: using pre_act_density may results in worse quality !!')
        if self.in_act_density:
            print('dvgo: using in_act_density may results in worse quality !!')

        # determine based grid resolution
        self.num_voxels_base = deform_num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # # determine the density bias shift
        # self.alpha_init = alpha_init
        # self.act_shift = np.log(1/(1-alpha_init) - 1)
        # print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(deform_num_voxels)

        # init occlusion voxel grid
        self.occlusion = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        # init color representation
        self.deformnet_kwargs = {
            'deformnet_dim': deformnet_dim,
            'deformnet_full_implicit': deformnet_full_implicit,
            'deformnet_depth': deformnet_depth, 'deformnet_width': deformnet_width,
            'posbase_pe': posbase_pe, 'timebase_pe': timebase_pe, 'deformnet_output': deformnet_output,
        }
        self.deformnet_full_implicit = deformnet_full_implicit
        self.deformnet_output = deformnet_output

        # feature voxel grid + shallow MLP  (fine stage)
        if self.deformnet_full_implicit:
            self.k0_dim = 0
        else:
            self.k0_dim = deformnet_dim
        self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        # torch.nn.init.xavier_normal_(self.k0)
        # self.k0 = torch.nn.Parameter(self.get_grid_worldcoords3().permute(3, 0, 1, 2).unsqueeze(0))
        self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('timefreq', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        dim0 = (3+3*posbase_pe*2) + (1+timebase_pe*2)
        if self.deformnet_full_implicit:
            pass
        else:
            dim0 += self.k0_dim * (1+timebase_pe*2)
        self.deformnet = nn.Sequential(
            nn.Linear(dim0, deformnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(deformnet_width, deformnet_width), nn.ReLU(inplace=True))
                for _ in range(deformnet_depth-2)
            ],
            nn.Linear(deformnet_width, self.deformnet_output),
        )
        nn.init.constant_(self.deformnet[-1].bias, 0)
        self.deformnet[-1].weight.data *= 0.0
        print('dvgo: feature voxel grid', self.k0.shape)
        print('dvgo: mlp', self.deformnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)

        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        self.mask_cache = None
        self.nonempty_mask = None
        self.train_times = train_times

        if mask_cache_path is not None:
            # mask cache
            print('mask cache path: ', mask_cache_path)
            self.mask_cache = MaskCacheDeform(
                    path=mask_cache_path,
                    train_times=train_times,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self._set_nonempty_mask()

            # reload grid and network
            cache_model = torch.load(mask_cache_path)
            cache_model_occlusion = cache_model['model_state_dict']['deformgrid.occlusion']
            cache_model_k0 = cache_model['model_state_dict']['deformgrid.k0']
            cache_xyz_min = torch.FloatTensor(cache_model['MaskCache_kwargs']['xyz_min']).to(cache_model_k0.device)
            cache_xyz_max = torch.FloatTensor(cache_model['MaskCache_kwargs']['xyz_max']).to(cache_model_k0.device)

            grid_xyz = self.get_grid_worldcoords3().unsqueeze(0)

            ind_norm = ((grid_xyz - cache_xyz_min) / (cache_xyz_max - cache_xyz_min)).flip((-1,)) * 2 - 1

            self.occlusion = torch.nn.Parameter(
                F.grid_sample(cache_model_occlusion, ind_norm, align_corners=True))

            if self.k0_dim > 0:
                self.k0 = torch.nn.Parameter(
                    F.grid_sample(cache_model_k0, ind_norm, align_corners=True))

            # load deformnet weights
            dn_static_dict = self.deformnet.state_dict()
            for k, v in dn_static_dict.items():
                if 'deformgrid.deformnet.' + k in cache_model['model_state_dict'].keys():
                    v = cache_model['model_state_dict']['deformgrid.deformnet.' + k]
                    dn_static_dict.update({k: v})
            self.deformnet.load_state_dict(dn_static_dict)
        print("--------------- Finish ----------------")

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'deform_num_voxels': self.num_voxels,
            'deform_num_voxels_base': self.num_voxels_base,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_deform_thres': self.fast_deform_thres,
            'train_times': self.train_times,
            **self.deformnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.occlusion.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.occlusion.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.occlusion.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        self.occlusion[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.occlusion.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.occlusion.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.occlusion.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.occlusion[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.occlusion = torch.nn.Parameter(
            F.interpolate(self.occlusion.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.occlusion.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.occlusion.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.occlusion).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def k0_total_variation(self):
        v = self.k0
        return total_variation(v, self.nonempty_mask)

    def occlusion_mean(self):
        return torch.mean(torch.sigmoid(self.occlusion))

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.k0.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox

    def get_grid_worldcoords(self,):
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.range(0, self.world_size[0]-1),
            torch.range(0, self.world_size[1]-1),
            torch.range(0, self.world_size[2]-1),
            indexing='ij'
        )
        grid_coord = torch.stack([grid_z, grid_y, grid_x], dim=-1) # grid_sample use pixel positions, inverse
        grid_coord = 2 * grid_coord / (self.world_size.flip((-1,)) - 1) - 1 # [-1 1]

        grid_coord = (grid_coord + 1) / 2
        grid_coord = grid_coord.flip((-1,))
        grid_coord = grid_coord * (self.xyz_max - self.xyz_min) + self.xyz_min

        return grid_coord

    def get_grid_worldcoords2(self,):
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.world_size[0]),
            torch.linspace(0, 1, self.world_size[1]),
            torch.linspace(0, 1, self.world_size[2]),
        ), -1)
        grid_coord = self.xyz_min * (1-interp) + self.xyz_max * interp

        return grid_coord

    def get_grid_worldcoords3(self,):
        grid_coord = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)

        return grid_coord

    def forward(self, rays_pts, timestep, mask_outbbox, **render_kwargs):
        '''
            give occlusion mask and deformation according to given positions
        '''

        # update mask for query points in known free space
        if self.mask_cache is not None:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        # ------------------- time a --------------------
        torch.cuda.synchronize()
        time_a = time.time()
        # -----------------------------------------------

        # query for occlusion mask
        occlusion_mask = torch.zeros_like(rays_pts[...,0])
        if self.pre_act_density:
            # pre-activation
            occlusion_mask[~mask_outbbox] = self.grid_sampler(
                    rays_pts[~mask_outbbox], torch.sigmoid(self.occlusion))
        elif self.in_act_density:
            # in-activation : same with pre-activation in terms of occlusion mask
            occlusion_mask[~mask_outbbox] = self.grid_sampler(
                    rays_pts[~mask_outbbox], torch.sigmoid(self.occlusion))
        else:
            # post-activation
            occlusion_feat = self.grid_sampler(rays_pts[~mask_outbbox], self.occlusion)
            occlusion_mask[~mask_outbbox] = torch.sigmoid(occlusion_feat)


        # ------------------- time b --------------------
        torch.cuda.synchronize()
        time_b = time.time()
        # -----------------------------------------------

        # query for deform
        # mask = (occlusion_mask > self.fast_deform_thres)
        mask = ~mask_outbbox
        k0 = torch.zeros(*occlusion_mask.shape, self.k0_dim).to(occlusion_mask)
        if not self.deformnet_full_implicit:
            k0[mask] = self.grid_sampler(rays_pts[mask], self.k0)

        k0_view = k0

        # ------------------- time c --------------------
        torch.cuda.synchronize()
        time_c = time.time()
        # -----------------------------------------------

        timestep_emb = (timestep.unsqueeze(-1) * self.timefreq).flatten(-2)
        timestep_emb = torch.cat([timestep, timestep_emb.sin(), timestep_emb.cos()], -1)

        # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
        rays_xyz = rays_pts[mask]
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        k0_view_mask = k0_view[mask]
        k0_emb = (k0_view_mask.unsqueeze(-1) * self.timefreq).flatten(-2)
        k0_emb = torch.cat([k0_view_mask, k0_emb.sin(), k0_emb.cos()], -1)

        deform_feat = torch.cat([
            k0_emb,
            xyz_emb,
            # TODO: use `rearrange' to make it readable
            timestep_emb.flatten(0,-2).unsqueeze(-2).repeat(1,occlusion_mask.shape[-1],1)[mask.flatten(0,-2)]
        ], -1)

        deform_logit = torch.zeros(*occlusion_mask.shape, self.deformnet_output).to(occlusion_mask)
        if self.deformnet_output > 3:
            deform_logit[..., -1] = 100000

        deform_logit[mask] = self.deformnet(deform_feat)
        if self.deformnet_output > 3:
            deform = deform_logit[..., :-1]
            occlusion_mask = deform_logit[..., -1, None]
            occlusion_mask = torch.sigmoid(occlusion_mask)
        else:
            deform = deform_logit

        # ------------------- time d --------------------
        torch.cuda.synchronize()
        time_d = time.time()

        time_dict = {
            'query_occ': time_b - time_a,
            'query_k0': time_c - time_b,
            'query_dnet': time_d - time_c
        }
        # -----------------------------------------------

        return  occlusion_mask, deform, mask, time_dict


class VoxRendererDynamic(torch.nn.Module):
    def __init__(
        self,
        deformgrid,
        radiancegrid,
        **kwargs
    ):
        super(VoxRendererDynamic, self).__init__()
        self.deformgrid = deformgrid
        self.radiancegrid = radiancegrid

    def scale_volume_grid(self, factor):
        self.deformgrid.scale_volume_grid(self.deformgrid.num_voxels * factor)
        self.radiancegrid.scale_volume_grid(self.radiancegrid.num_voxels * factor)


    def get_deform_grid(self, time_step):
        grid_coord = self.deformgrid.get_grid_worldcoords3()
        grid_coord = grid_coord.reshape(1, -1, 3)
        num_grid = grid_coord.shape[1]

        timesstep = torch.ones(1, 1) * time_step
        mask_outbbox = torch.zeros(1, num_grid) > 0
        occ, deformation, _, _ = self.deformgrid(grid_coord, timesstep, mask_outbbox)

        can_mask = timesstep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deformation.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        deformation[can_mask] = 0.
        grid_coord += deformation

        return grid_coord, occ

    def get_deform_alpha_rgb(self, time_step):
        grid_coord, occ = self.get_deform_grid(time_step)
        densities = self.radiancegrid.grid_sampler(grid_coord.reshape(-1,3), self.radiancegrid.density)
        alpha = self.radiancegrid.activate_density(densities)
        alpha = alpha.reshape([1, 1, *self.deformgrid.world_size])
        occ = occ.reshape([1, 1, *self.deformgrid.world_size])

        k0 = self.radiancegrid.grid_sampler(grid_coord.reshape(-1,3), self.radiancegrid.k0)
        rgb = torch.sigmoid(k0).permute(1,0)
        rgb = rgb.reshape([1, 3, *self.deformgrid.world_size])
        return alpha, rgb, occ

    def forward(self, rays_o, rays_d, timestep, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering'''
        ret_dict = {}

        # ------------------- time a --------------------
        torch.cuda.synchronize()
        time_a = time.time()
        # -----------------------------------------------

        # sample points on rays
        rays_pts, mask_outbbox = self.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.deformgrid.voxel_size_ratio

        # ------------------- time b --------------------
        torch.cuda.synchronize()
        time_b = time.time()
        # -----------------------------------------------

        # inference deformation
        # if torch.mean(timestep) > 0.:
        #     occlusion_mask, deform, mask = self.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        #     occlusion_mask = occlusion_mask.unsqueeze(-1)
        #     rays_pts += deform * occlusion_mask
        # else:
        #     deform = torch.zeros_like(rays_pts)
        #     occlusion_mask = torch.zeros_like(rays_pts)[..., 0, None]

        occlusion_mask, deform, mask_d, time_dict = self.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        # deform = deform * occlusion_mask.unsqueeze(-1)
        if 'bg_points_sel' in render_kwargs.keys():
            bg_mask_outbbox = torch.ones(render_kwargs['bg_points_sel'].shape[0], 1).to(rays_pts.device) > 0
            bg_time_step = timestep[:render_kwargs['bg_points_sel'].shape[0]]
            _, bg_points_deform, _, _ = self.deformgrid(render_kwargs['bg_points_sel'].unsqueeze(-2), bg_time_step, bg_mask_outbbox, **render_kwargs)
            ret_dict.update({'bg_points_delta': bg_points_deform})

        can_mask = timestep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        can_loss = torch.mean(torch.abs(deform[can_mask]))
        deform[can_mask] = 0.

        # deform *= 0

        rays_pts += deform

        # ------------------- time c --------------------
        torch.cuda.synchronize()
        time_c = time.time()
        # -----------------------------------------------

        # inference alpha, rgb
        if self.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None

        alpha, alphainv_cum, rgb, weights, mask_r = self.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)

        # ------------------- time d --------------------
        torch.cuda.synchronize()
        time_d = time.time()
        # -----------------------------------------------

        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_loss': can_loss,
            'can_mask': can_mask
        })

        # ------------------- time e --------------------
        torch.cuda.synchronize()
        time_e = time.time()

        time_dict.update({
            'sample_pts': time_b - time_a,
            'deform_forward': time_c - time_b,
            'rad_forward': time_d - time_c,
            'render': time_e - time_d
        })
        ret_dict.update({
            'time_dict': time_dict
        })
        # -----------------------------------------------

        return ret_dict


''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, ks=3):
        super().__init__()
        st = torch.load(path)
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(st['MaskCache_kwargs']['xyz_min']))
        self.register_buffer('xyz_max', torch.FloatTensor(st['MaskCache_kwargs']['xyz_max']))
        if 'radiancegrid.density' in st['model_state_dict'].keys():
            den_key = 'radiancegrid.density'
        else:
            den_key = 'density'
        self.register_buffer('density', F.max_pool3d(
            st['model_state_dict'][den_key], kernel_size=ks, padding=ks//2, stride=1))
        self.act_shift = st['MaskCache_kwargs']['act_shift']
        self.voxel_size_ratio = st['MaskCache_kwargs']['voxel_size_ratio']
        self.nearest = st['MaskCache_kwargs'].get('nearest', False)
        self.pre_act_density = st['MaskCache_kwargs'].get('pre_act_density', False)
        self.in_act_density = st['MaskCache_kwargs'].get('in_act_density', False)

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if self.nearest:
            density = F.grid_sample(self.density, ind_norm, align_corners=True, mode='nearest')
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        elif self.pre_act_density:
            alpha = 1 - torch.exp(-F.softplus(self.density + self.act_shift) * self.voxel_size_ratio)
            alpha = F.grid_sample(self.density, ind_norm, align_corners=True)
        elif self.in_act_density:
            density = F.grid_sample(F.softplus(self.density + self.act_shift), ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-density * self.voxel_size_ratio)
        else:
            density = F.grid_sample(self.density, ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        alpha = alpha.reshape(*shape)
        return (alpha >= self.mask_cache_thres)


class MaskCacheDeform(nn.Module):
    def __init__(self, path, mask_cache_thres, train_times, ks=3):
        super().__init__()
        print('dvgo: making cache, start')
        self.mask_cache_thres = mask_cache_thres
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = utils.load_model_dynamic(
            [VoxRendererDynamic, DeformVoxGO, DirectVoxGO],
            path
            ).to(device)
        self.xyz_max = model.deformgrid.xyz_max
        self.xyz_min = model.deformgrid.xyz_min

        alphas =[]
        with torch.no_grad():
            for i in range(0, len(train_times), 1):
                # print('dvgo:  making cache, processing time step: ', train_times[i])
                ti = train_times[i]
                alpha, _, _ = model.get_deform_alpha_rgb(ti)
                alphas.append(alpha)
        self.alpha, _ = torch.max(torch.stack(alphas, dim=-1), dim=-1)
        self.alpha = F.max_pool3d(
            self.alpha, kernel_size=ks, padding=ks//2, stride=1)
        del model
        print('dvgo:  making cache, finished')

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        al = F.grid_sample(self.alpha, ind_norm, align_corners=True)
        al = al.reshape(*shape)
        return (al >= self.mask_cache_thres)


''' Misc
'''
def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[...,[0]]), p.clamp_min(1e-10).cumprod(-1)], -1)

def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1-alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum

def total_variation4(v, mask=None):
    mask = None
    tv2 = v.diff(dim=2, n=1).abs()
    tv3 = v.diff(dim=3, n=1).abs()
    tv4 = v.diff(dim=4, n=1).abs()
    if mask is not None:
        tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
        tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
        tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

def total_variation(v, mask=None):
    mask = None
    tv2 = v.diff(dim=2, n=2).abs().mean(dim=1, keepdim=True)
    tv3 = v.diff(dim=3, n=2).abs().mean(dim=1, keepdim=True)
    tv4 = v.diff(dim=4, n=2).abs().mean(dim=1, keepdim=True)
    if mask is not None:
        maska = mask[:,:,:-1] & mask[:,:,1:]
        tv2 = tv2[maska[:,:,:-1] & maska[:,:,1:]]
        maskb = mask[:,:,:,:-1] & mask[:,:,:,1:]
        tv3 = tv3[maskb[:,:,:,:-1] & maskb[:,:,:,1:]]
        maskc = mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]
        tv4 = tv4[maskc[:,:,:,:,:-1] & maskc[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

def total_variation2(v, mask=None):
    tv2 = torch.square(v.diff(dim=2, n=2))
    tv3 = torch.square(v.diff(dim=3, n=2))
    tv4 = torch.square(v.diff(dim=4, n=2))
    if mask is not None:
        tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
        tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
        tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def get_training_rays_dynamic(rgb_tr, times_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    times_tr = times_tr.reshape(-1,1,1,1).expand(-1, rgb_tr.shape[1], rgb_tr.shape[2], -1)
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = model.sample_ray(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i+CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling_dynamic(rgb_tr_ori, times_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    times_tr_ori = times_tr_ori.reshape(-1,1,1,1).expand(-1, rgb_tr_ori.shape[1], rgb_tr_ori.shape[2], -1)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    times_tr = torch.zeros([N,1], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, tim, (H, W), K in zip(train_poses, rgb_tr_ori, times_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = model.radiancegrid.sample_ray(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.radiancegrid.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i+CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        times_tr[top:top+n].copy_(tim[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    times_tr = times_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

