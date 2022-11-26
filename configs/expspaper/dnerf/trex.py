_base_ = './settings.py'

expname = 'ndvg_dnerf_trex'

data = dict(
    datadir='./data/dnerf_synthetic/trex',
    dataset_type='dnerf',
    white_bkgd=True,
)
