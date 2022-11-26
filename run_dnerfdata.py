import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
from tqdm.contrib import tzip

import mmcv
import cv2
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import ndvg_renderer, utils
from lib.load_data import load_data_dynamic

from torch.utils.tensorboard import SummaryWriter

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_debug", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # model options
    parser.add_argument("--use_coarse_voxgrid", action='store_true',
                        help='use coarse voxel grid model')
    parser.add_argument("--use_fine_voxgrid", action='store_true',
                        help='use fine voxel grid model')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_log",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, render_times, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, gt_mask=None, savedir=None, render_factor=0,
                      eval_ssim=True, eval_lpips_alex=False, eval_lpips_vgg=True,
                      pre=''):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    disps = []
    err_maps = []
    mse = []
    psnrs = []
    psnrs_mask = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, (c2w, timestep) in enumerate(tzip(render_poses, render_times)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = ndvg_renderer.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        timestep = timestep.reshape(1,1,-1).expand(rays_o.shape[0], rays_o.shape[1], -1)
        keys = ['rgb_marched', 'disp']
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, ts, vd, **render_kwargs).items() if k in keys}
            for ro, rd, ts, vd in zip(rays_o.split(16, 0), rays_d.split(16, 0),  timestep.split(16, 0), viewdirs.split(16, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        disp = render_result['disp'].cpu().numpy()

        rgbs.append(rgb)
        disps.append(disp)
        if i==0:
            print('Testing', rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor==0:
            mask = gt_mask[i] > 0
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            p_mask = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])[mask[...,0]]))
            psnrs.append(p)
            psnrs_mask.append(p_mask)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
            mse.append(np.mean(np.square(rgb - gt_imgs[i])))
            error_map = np.clip(np.mean(np.square(rgb - gt_imgs[i]), axis=-1), a_min=0, a_max=0.005)/0.005
            error_map = (255 * error_map).astype(np.uint8)
            error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET) # (3, H, W)
            error_map = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)
            err_maps.append(error_map)
            print('PSNR ', i, ' : ', p)

        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            filename = os.path.join(savedir, pre + '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            disp8 = utils.to8b(disps[-1] / np.max(disps[-1]))
            filename = os.path.join(savedir, pre + 'disp_{:03d}.png'.format(i))
            imageio.imwrite(filename, disp8)
            if len(err_maps) > 0:
                rgb8 = utils.to8b(err_maps[-1])
                filename = os.path.join(savedir, pre + '_error_{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        '''
        print('Testing psnr', [f'{p:.3f}' for p in psnrs])
        if eval_ssim: print('Testing ssim', [f'{p:.3f}' for p in ssims])
        if eval_lpips_vgg: print('Testing lpips (vgg)', [f'{p:.3f}' for p in lpips_vgg])
        if eval_lpips_alex: print('Testing lpips (alex)', [f'{p:.3f}' for p in lpips_alex])
        '''
        print('Testing MSE', np.mean(mse), '(avg)')
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing psnr masked', np.mean(psnrs_mask), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    return rgbs, disps

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data_dynamic(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'render_times', 'images', 'times', 'mask'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    data_dict['times'] = torch.Tensor(data_dict['times'])
    return data_dict

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = ndvg_renderer.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo_deform(model_class, model_path, thres, times_one):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xyz_min = []
    xyz_max = []
    model = utils.load_model_dynamic(model_class, model_path).to(device)

    dense_xyz = model.deformgrid.get_grid_worldcoords3()

    with torch.no_grad():
        for i in range(0, times_one.shape[0], 1):
            ti = times_one[i]
            alpha, _, _ = model.get_deform_alpha_rgb(ti)
            mask = (alpha.squeeze() > thres)
            active_xyz = dense_xyz[mask]
            xyz_min.append(active_xyz.amin(0))
            xyz_max.append(active_xyz.amax(0))
            print('compute_bbox_by_coarse_geo: processed deform time ', ti,
                ' ', active_xyz.amin(0), ' ', active_xyz.amax(0))

    xyz_min = torch.stack(xyz_min, dim=0)
    xyz_max = torch.stack(xyz_max, dim=0)
    xyz_min = xyz_min.amin(0)
    xyz_max = xyz_max.amax(0)

    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None, tb_writer=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, render_times, images, times = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'render_times', 'images', 'times'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    deform_num_voxels = model_kwargs.pop('deform_num_voxels')
    if len(cfg_train.pg_scale) and reload_ckpt_path is None:
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
        deform_num_voxels = int(deform_num_voxels / (2**len(cfg_train.pg_scale)))
    model = ndvg_renderer.VoxRendererDynamic(
        ndvg_renderer.DeformVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            deform_num_voxels=deform_num_voxels,
            mask_cache_path=coarse_ckpt_path,
            train_times=data_dict['times'][data_dict['i_train']],
            **model_kwargs
        ),
        ndvg_renderer.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs
        )
    )

    if cfg_model.maskout_near_cam_vox:
        model.radiancegrid.maskout_near_cam_vox(poses[i_train,:3,3], near)
        model.deformgrid.maskout_near_cam_vox(poses[i_train,:3,3], near)
    model = model.to(device)

    # init optimizer
    optimizer = utils.create_optimizer_or_freeze_model_dynamic(model, cfg_train, global_step=0)

    # load checkpoint if there is
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = ndvg_renderer.get_training_rays_in_maskcache_sampling_dynamic(
                    rgb_tr_ori=rgb_tr_ori,
                    times_tr_ori=times[i_train],
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = ndvg_renderer.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = ndvg_renderer.get_training_rays_dynamic(
                rgb_tr=rgb_tr_ori,
                times_tr=times[i_train],
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = ndvg_renderer.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.radiancegrid.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.radiancegrid.density[cnt <= 2] = -100
        per_voxel_init()

    # GOGO
    time_log = {'sample_rays': 0, 'forward': 0, 'loss': 0, 'backward': 0, 'optimize': 0,
                'sample_pts': 0, 'deform_forward': 0, 'rad_forward': 0, 'render': 0,
                'query_occ': 0, 'query_k0': 0, 'query_dnet': 0,
                }
    torch.cuda.empty_cache()
    psnr_lst = []
    torch.cuda.synchronize()
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            model.scale_volume_grid(2)
            optimizer = utils.create_optimizer_or_freeze_model_dynamic(model, cfg_train, global_step=0)
            model.radiancegrid.density.data.sub_(1)
            model.deformgrid.occlusion.data.sub_(1)

        # ------------------- time a --------------------
        torch.cuda.synchronize()
        time_a = time.time()
        # -----------------------------------------------

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            times_step = times_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            if cfg_train.progressive_training:
                max_train_num = global_step // cfg_train.num_persample # linear
                max_train_num = min(max(int(max_train_num), cfg_train.min_sample), rgb_tr.shape[0])
            else:
                max_train_num = rgb_tr.shape[0]
            if cfg_train.sample_oneimage:
                sel_b = torch.ones(cfg_train.N_rand).to(torch.int64)
                sel_b = sel_b * torch.randint(max_train_num, [1])[0]
                if global_step % 5 == 0:
                    sel_b = sel_b * 0
            else:
                sel_b = torch.randint(max_train_num, [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            times_step = times_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            times_step = times_step.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # ------------------- time b --------------------
        torch.cuda.synchronize()
        time_b = time.time()
        # -----------------------------------------------

        # volume rendering
        render_result = model(rays_o, rays_d, times_step, viewdirs, global_step=global_step, **render_kwargs)

        # ------------------- time c --------------------
        torch.cuda.synchronize()
        time_c = time.time()
        # -----------------------------------------------

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach()).item()
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][...,-1].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += cfg_train.weight_rgbper * rgbper_loss

        if cfg_train.weight_tv_density>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_density * model.radiancegrid.density_total_variation()
        if cfg_train.weight_tv_k0>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_k0 * model.radiancegrid.k0_total_variation()

        if cfg_train.deform_weight_tv_k0>0 and global_step>cfg_train.deform_tv_from and global_step%cfg_train.deform_tv_every==0 and not cfg_model.deformnet_full_implicit:
            dtv = model.deformgrid.k0_total_variation()
            loss += cfg_train.deform_weight_tv_k0 * dtv
        else:
            if not cfg_model.deformnet_full_implicit:
                dtv = model.deformgrid.k0_total_variation()
            else:
                dtv = torch.zeros(1)
        if cfg_train.weight_deform_norm>0:
            dn = torch.mean(torch.abs(render_result['deformation']))
            loss += cfg_train.weight_deform_norm * dn
        else:
            dn = torch.zeros(1)

        if cfg_train.deform_weight_occ>0:
            dwo = torch.mean(1 - render_result['occlusion'])
            loss += cfg_train.deform_weight_occ * dwo
        else:
            dwo = torch.mean(1 - render_result['occlusion'])

        if torch.sum(render_result['can_mask']) > 0:
            can_loss = render_result['can_loss']
            loss += 1000. * can_loss
        else:
            can_loss = torch.zeros(1)

        # ------------------- time d --------------------
        torch.cuda.synchronize()
        time_d = time.time()
        # -----------------------------------------------

        loss.backward()

        # ------------------- time e --------------------
        torch.cuda.synchronize()
        time_e = time.time()
        # -----------------------------------------------

        optimizer.step()

        # ------------------- time f --------------------
        torch.cuda.synchronize()
        time_f = time.time()
        # -----------------------------------------------

        # ------------------- time b --------------------
        time_log['sample_rays'] += (time_b - time_a)
        time_log['forward'] += (time_c - time_b)
        time_log['loss'] += (time_d - time_c)
        time_log['backward'] += (time_e - time_d)
        time_log['optimize'] += (time_f - time_e)
        sr_time = time_log['sample_rays']/(global_step+1)
        f_time = time_log['forward']/(global_step+1)
        l_time = time_log['loss']/(global_step+1)
        b_time = time_log['backward']/(global_step+1)
        o_time = time_log['optimize']/(global_step+1)

        time_log['sample_pts'] += render_result['time_dict']['sample_pts']
        time_log['deform_forward'] += render_result['time_dict']['deform_forward']
        time_log['rad_forward'] += render_result['time_dict']['rad_forward']
        time_log['render'] += render_result['time_dict']['render']
        sp_time = time_log['sample_pts']/(global_step+1)
        df_time = time_log['deform_forward']/(global_step+1)
        rf_time = time_log['rad_forward']/(global_step+1)
        r_time = time_log['render']/(global_step+1)

        time_log['query_occ'] += render_result['time_dict']['query_occ']
        time_log['query_k0'] += render_result['time_dict']['query_k0']
        time_log['query_dnet'] += render_result['time_dict']['query_dnet']
        qo_time = time_log['query_occ']/(global_step+1)
        qk_time = time_log['query_k0']/(global_step+1)
        qn_time = time_log['query_dnet']/(global_step+1)
        # -----------------------------------------------

        psnr_lst.append(psnr)

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_log==0 and tb_writer is not None:
            tb_writer.add_scalar(stage+"_Loss", loss.item(), global_step)
            tb_writer.add_scalar(stage+"_PSNR", np.mean(psnr_lst), global_step)
            tb_writer.add_scalar(stage+"_DN", dn.item(), global_step)
            tb_writer.add_scalar(stage+"_DWO", dwo.item(), global_step)
            tb_writer.add_scalar(stage+"_DTV", dtv.item(), global_step)
            tb_writer.add_scalar(stage+"_CAN", can_loss.item(), global_step)
            def plot_gridfeature(writer, k0, prex):
                k0shape = list(k0.shape[-3:])
                def plot_feature(feature): # [N, H, W]
                    feat_dim = feature.shape[0]
                    imgs = []
                    for i in range(feat_dim//3):
                        img = feature[3*i:3*(i+1), :, :]
                        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img) + 1e-8)
                        imgs.append(img)
                    imgs = torch.cat(imgs, dim=-1)
                    return imgs
                imgs = []
                f1 = k0[0, :, k0shape[0]//2, :, :]
                f2 = k0[0, :, :, k0shape[1]//2, :]
                f3 = k0[0, :, :, :, k0shape[2]//2]
                imgs.append(plot_feature(f1))
                imgs.append(plot_feature(f2))
                imgs.append(plot_feature(f3))
                writer.add_image(prex+'_k0_vis_slice0', imgs[0], global_step, dataformats='CHW')
                writer.add_image(prex+'_k0_vis_slice1', imgs[1], global_step, dataformats='CHW')
                writer.add_image(prex+'_k0_vis_slice2', imgs[2], global_step, dataformats='CHW')
            if not cfg_model.deformnet_full_implicit:
                plot_gridfeature(tb_writer, model.deformgrid.k0, stage+'_deform')
            if not cfg_model.rgbnet_full_implicit:
                plot_gridfeature(tb_writer, model.radiancegrid.k0, stage+'_radiance')

        if stage == 'fine' and global_step > 9999 and global_step%5000==0:
            print('test for step: ', global_step)
            with torch.no_grad():
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{stage}_{global_step:05d}')
                os.makedirs(testsavedir, exist_ok=True)
                if stage == 'coarse':
                    stepsize = cfg.coarse_model_and_render.stepsize
                else:
                    stepsize = cfg.fine_model_and_render.stepsize
                render_viewpoints_kwargs = {
                    'model': model,
                    'ndc': cfg.data.ndc,
                    'render_kwargs': {
                        'near': data_dict['near'],
                        'far': data_dict['far'],
                        'bg': 1 if cfg.data.white_bkgd else 0,
                        'stepsize': stepsize,
                        'inverse_y': cfg.data.inverse_y,
                        'flip_x': cfg.data.flip_x,
                        'flip_y': cfg.data.flip_y,
                    },
                }
                rgbs, disps = render_viewpoints(
                        render_poses=data_dict['poses'][data_dict['i_test']],
                        render_times=data_dict['times'][data_dict['i_test']],
                        HW=data_dict['HW'][data_dict['i_test']],
                        Ks=data_dict['Ks'][data_dict['i_test']],
                        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                        gt_mask=data_dict['mask'][data_dict['i_test']],
                        savedir=testsavedir,
                        eval_ssim=True, eval_lpips_alex=False, eval_lpips_vgg=True,
                        **render_viewpoints_kwargs)


        if global_step%args.i_print==0:
            torch.cuda.synchronize()
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write('--------------------------------------------------------------- \n'
                       f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.6f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'dn: {dn.item():.6f} / dwo: {dwo.item():.6f} / dtv: {dtv.item():.6f} / can: {can_loss.item():.6f} / '
                       f'Eps: {eps_time_str} \n'
                       f'SampleRays: {sr_time:.6f} / Forward: {f_time:.6f} / Loss: {l_time:.6f} / Backward: {b_time:.6f} / Optimize: {o_time:.6f} \n'
                       f'SamplePts: {sp_time:.6f} / DeformFor: {df_time:.6f} / RadFor: {rf_time:.6f} / Render: {r_time:.6f} \n'
                       f'QueryOcc: {qo_time:.6f} / Queryk0: {qk_time:.6f} / QueryDNet: {qn_time:.6f}'
                       )
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'radiance_kwargs': model.radiancegrid.get_kwargs(),
                'MaskCache_kwargs': model.radiancegrid.get_MaskCache_kwargs(),
                'deform_kwargs': model.deformgrid.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'radiance_kwargs': model.radiancegrid.get_kwargs(),
            'MaskCache_kwargs': model.radiancegrid.get_MaskCache_kwargs(),
            'deform_kwargs': model.deformgrid.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    torch.cuda.synchronize()
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # tensorboard
    writer = SummaryWriter(os.path.join(cfg.basedir, cfg.expname))

    # coarse geometry searching
    if args.use_coarse_voxgrid:
        eps_coarse = time.time()
        xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse', tb_writer=writer)
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)

    if args.use_fine_voxgrid:
        # fine detail reconstruction
        eps_fine = time.time()
        if args.use_coarse_voxgrid:
            coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
            xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo_deform(
                    model_class=[ndvg_renderer.VoxRendererDynamic, ndvg_renderer.DeformVoxGO, ndvg_renderer.DirectVoxGO],
                    model_path=coarse_ckpt_path,
                    thres=cfg.fine_model_and_render.bbox_thres,
                    times_one=data_dict['times'][data_dict['i_train']],
            )
        else:
            coarse_ckpt_path = None
            xyz_min_fine, xyz_max_fine = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                data_dict=data_dict, stage='fine',
                coarse_ckpt_path=coarse_ckpt_path,
                tb_writer=writer)
        eps_fine = time.time() - eps_fine
        eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
        print('train: fine detail reconstruction in', eps_time_str)

    torch.cuda.synchronize()
    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ', which is', eps_time, ' s)')


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.render_video_debug:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            if args.use_fine_voxgrid:
                print('basedir: ', cfg.basedir)
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
                stepsize = cfg.fine_model_and_render.stepsize
            else:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
                stepsize = cfg.coarse_model_and_render.stepsize
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model = utils.load_model_dynamic(
            [ndvg_renderer.VoxRendererDynamic, ndvg_renderer.DeformVoxGO, ndvg_renderer.DirectVoxGO],
            ckpt_path
        ).to(device)
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

    # render trainset and eval
    if args.render_train:
        print("Rendering train images")
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                render_times=data_dict['times'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                gt_mask =data_dict['mask'][data_dict['i_train']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        print("Rendering test images")
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                render_times=data_dict['times'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                gt_mask=data_dict['mask'][data_dict['i_test']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        print("Rendering videos")
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        poses_input = data_dict['render_poses'].clone()
        # poses_input[:] = poses_input[27]
        times_input =  data_dict['render_times'].clone() * 0
        # times_input=torch.linspace(0., 1.,6)
        rgbsfixt, dispsfixt = render_viewpoints(
                render_poses=poses_input,
                render_times=times_input,
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                pre='view27',
                **render_viewpoints_kwargs)
        poses_input = data_dict['render_poses'].clone()
        # poses_input[:] = poses_input[11]
        times_input =  data_dict['render_times'].clone()
        # times_input=torch.linspace(0., 1.,6)
        rgbsv, dispsv = render_viewpoints(
                render_poses=poses_input,
                render_times=times_input,
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                pre='view11',
                **render_viewpoints_kwargs)
        rgbs = np.concatenate((rgbsfixt, rgbsv), 0)
        disps = np.concatenate((dispsfixt, dispsv), 0)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video_debug:
        print("Rendering videos")
        debug_idx = 33
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_debug_{debug_idx:2d}_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        render_poses = data_dict['render_poses'].clone()
        render_times = data_dict['render_times'].clone()
        render_poses[:] = render_poses[debug_idx]
        rgbsfixt, dispsfixt = render_viewpoints(
                render_poses=render_poses,
                render_times=render_times,
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                pre='fixview',
                **render_viewpoints_kwargs)
        render_poses = data_dict['render_poses'].clone()
        render_times = data_dict['render_times'].clone()
        render_times[:] = render_times[debug_idx]
        rgbsv, dispsv = render_viewpoints(
                render_poses=render_poses,
                render_times=render_times,
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                pre='fixtime',
                **render_viewpoints_kwargs)
        rgbs = np.concatenate((rgbsfixt, rgbsv), 0)
        disps = np.concatenate((dispsfixt, dispsv), 0)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)


    print('Done')

