_base_ = './settings.py'

expname = 'ndvg_dnerf_standup'

data = dict(
    datadir='./data/dnerf_synthetic/standup',
    dataset_type='dnerf',
    white_bkgd=True,
)
