_base_ = './settings.py'

expname = 'ndvg_dnerf_lego'

data = dict(
    datadir='./data/dnerf_synthetic/lego',
    dataset_type='dnerf',
    white_bkgd=True,
)
