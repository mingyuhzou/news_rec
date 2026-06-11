import os
from config.common import cfg as common
from config.model.sid_generate import cfg as model_cfg
cfg={
    'model_weight_path':os.path.join(model_cfg['ckpt_dir'],'Jun-08-2026_11-02-38/best_collision_model.pth'),
    'output_file':os.path.join(common['data_path'],'rqvae.npy'),
}