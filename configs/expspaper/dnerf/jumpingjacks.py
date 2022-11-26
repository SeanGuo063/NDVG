_base_ = './settings.py'

expname = 'ndvg_dnerf_jumpingjacks'

data = dict(
    datadir='./data/dnerf_synthetic/jumpingjacks',
    dataset_type='dnerf',
    white_bkgd=True,
)
