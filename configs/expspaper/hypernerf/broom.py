_base_ = './settings.py'

expname = 'ndvg_hypernerf_broom'

data = dict(
    datadir='./data/Hypernerf/broom2',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)
