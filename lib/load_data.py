import numpy as np

from .load_dnerf import load_blender_dnerf
from .load_hyper import load_hyper_data


def load_data_dynamic(args):

    K, depths, times = None, None, None

    if args.dataset_type == 'dnerf':
        images, poses, times, render_poses, render_times, hwf, i_split = load_blender_dnerf(
            args.datadir, args.half_res, args.testskip, args.normtime,
        )
        print('Loaded dnerf', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.
        mask = images[...,-1:]

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'hyper_dataset':
        data_class = load_hyper_data(datadir=args.datadir,
                                    use_bg_points=args.use_bg_points, add_cam=args.add_cam)
        data_dict = dict(
            data_class=data_class,
            near=data_class.near, far=data_class.far,
            i_train=data_class.i_train, i_val=data_class.i_test, i_test=data_class.i_test,)
        return data_dict

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses, render_times=render_times,
        images=images, depths=depths, times=times,
        irregular_shape=irregular_shape, mask=mask
    )
    return data_dict

