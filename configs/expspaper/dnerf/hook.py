_base_ = './settings.py'

expname = 'ndvg_dnerf_hook'

data = dict(
    datadir='./data/dnerf_synthetic/hook',
    dataset_type='dnerf',
    white_bkgd=True,
)
