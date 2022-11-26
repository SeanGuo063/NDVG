_base_ = './settings.py'

expname = 'ndvg_hypernerf_peel-banana'

data = dict(
    datadir='./data/Hypernerf/vrig-peel-banana',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)
