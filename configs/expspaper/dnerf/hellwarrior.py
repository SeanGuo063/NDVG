_base_ = './settings.py'

expname = 'ndvg_dnerf_hellwarrior'

data = dict(
    datadir='./data/dnerf_synthetic/hellwarrior',
    dataset_type='dnerf',
    white_bkgd=True,
)
