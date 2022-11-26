_base_ = './settings.py'

expname = 'ndvg_hypernerf_chicken'

data = dict(
    datadir='./data/Hypernerf/vrig-chicken',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)
