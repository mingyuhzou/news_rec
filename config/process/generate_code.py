import os
from config.common import cfg as common
from config.model.generate_sid import cfg as model_cfg
cfg={
    'model_weight_path':os.path.join(model_cfg['ckpt_dir'],'best_collision_rate.pth'),
    'output_file':os.path.join(common['data_path'],'rqvae.npy'),
}