_base_ = '../../default.py'

basedir = './logs/dnerf'

data = dict(
    half_res=True
)

coarse_train = dict()

fine_train = dict()

coarse_model_and_render = dict()

fine_model_and_render = dict()