_base_ = '../../default.py'

basedir = './logs/hypernerf'

data = dict(
    half_res=True,
    use_bg_points=True,
    add_cam=True,
    load2gpu_on_the_fly=True,
    ndc=False
)

coarse_train = dict(
    progressive_training=True,     # whether use progressive training
    min_sample=10,                   # minimal number of samples to train
    num_persample=15,              # number of iterations per sample to increase
    pervoxel_lr=True,             # view-count-based lr
)

fine_train = dict(
    pervoxel_lr=True,             # view-count-based lr
)

coarse_model_and_render = dict()

fine_model_and_render = dict()