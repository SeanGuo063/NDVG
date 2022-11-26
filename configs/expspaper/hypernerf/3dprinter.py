_base_ = './settings.py'

expname = 'ndvg_hypernerf_3dprinter'

data = dict(
    datadir='./data/Hypernerf/vrig-3dprinter',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)
