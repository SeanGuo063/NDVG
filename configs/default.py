from copy import deepcopy

from numpy import False_

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    annot_path='',                # to support co3d
    split_path='',                # to support co3d
    sequence_name='',             # to support co3d
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=True,                # use half resolution of images
    factor=4,                     # [TODO]
    normtime=True,                # normalize times

    # Below are forward-facing llff specific settings. Not support yet.
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth

)

''' Template of training options
'''
coarse_train = dict(
    N_iters=5000,                # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_occlusion=1e-1,         # lr of occlusion voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lrate_deformnet=1e-3,         # lr of the mlp to estimate deformation
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    pervoxel_lr=False,             # view-count-based lr
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.01,     # weight of background entropy loss
    weight_rgbper=0.1,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_from=0,                    # count total variation loss from tv_from step
    weight_tv_density=0.0,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,             # weight of total variation loss of color/feature voxel grid
    pg_scale=[],                  # checkpoints for progressive scaling

    weight_deform_norm=1e-1,       # weight of deform loss
    progressive_training=True,   # whether use progressive training
    min_sample=10,                 # minimal number of samples to train
    num_persample=30,             # number of iterations per sample to increase
    deform_tv_every=1,                   # count total variation loss every tv_every step
    deform_tv_from=0,                    # count total variation loss from tv_from step
    deform_weight_tv_k0=1e0,             # weight of total variation loss of color/feature voxel grid
    sample_oneimage=False,                # for random sample, whether sample only in one image
    deform_weight_occ=0.          # weigth of occlusion reg loss

)

fine_train = deepcopy(coarse_train)
fine_train.update(dict(
    N_iters=10000,
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000],
    weight_deform_norm=1e-2,
    progressive_training=False,     # whether use progressive training
))

''' Template of model and rendering options
'''
coarse_model_and_render = dict(
    num_voxels=1024000,           # expected number of voxel
    num_voxels_base=1024000,      # to rescale delta distance
    deform_num_voxels=1664000,           # expected number of voxel
    deform_num_voxels_base=1664000,      # to rescale delta distance
    nearest=False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    fast_color_thres=0,           # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
    world_bound_scale=1,          # rescale the BBox enclosing the scene
    stepsize=0.5,                 # sampling stepsize in volume rendering

    deformnet_dim=4,             # dimension of feature for deformation in grid
    deformnet_full_implicit=False,# let the deform MLP ignore feature voxel grid
    deformnet_depth=4,            # depth of the deform MLP
    deformnet_width=64,          # width of the deform MLP
    fast_deform_thres=0.,         # threshold of occlusion value to skip the fine stage sampled point
    deformnet_output=4,           # dimension of output of the deform MLP
    app_code=False,               # whether to use appearance code for radiance grid
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    deform_num_voxels=190**3,
    deform_num_voxels_base=190**3,
    rgbnet_dim=12,
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
))

del deepcopy
