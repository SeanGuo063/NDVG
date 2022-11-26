_base_ = './settings.py'

expname = 'ndvg_dnerf_mutant'

data = dict(
    datadir='./data/dnerf_synthetic/mutant',
    dataset_type='dnerf',
    white_bkgd=True,
)
