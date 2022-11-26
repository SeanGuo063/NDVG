_base_ = './settings.py'

expname = 'ndvg_dnerf_bouncingballs'

data = dict(
    datadir='./data/dnerf_synthetic/bouncingballs',
    dataset_type='dnerf',
    white_bkgd=True,
)
