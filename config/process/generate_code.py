import os
from config.common import cfg as common
from config.model.rq_ave import cfg as model_cfg
cfg={
    'model_weight_path':os.path.join(model_cfg['ckpt_dir'],'Feb-12-2026_18-09-07/best_collision_model.pth'),
    'output_file':os.path.join(common['data_path'],'rqvae.npy'),
}