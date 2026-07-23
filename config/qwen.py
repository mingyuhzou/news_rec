import os

from config.common import cfg as common_config
from config.generate_code import cfg as generate_code
from config.generate_sid import cfg as rq_ave


cfg={

    'batch_size':256,
    'infer_size':96,
    'num_epochs':21,
    'lr':1e-4,
    'device':'cuda',
    'seed':2025,
    'early_stop':10,

    'hidden_size':256,
    'intermediate_size':1024,
    'num_hidden_layers':4,
    'num_attention_heads':4,

    'vocab_size':1025,
    'pad_token_id':0,
    'eos_token_id':1,

    'max_position_embeddings':2048,
    'num_key_value_heads':4,
    'max_len':100,

    'dataset_path':common_config['data_path'],

    'code_path':generate_code['output_file'],

    'mode':'train',

    'code_len':len(rq_ave['num_emb_list'])+1,# 加一是因为生成sid时会往后补一位解决冲突

    'beam_size':21,
    'log_path':
        os.path.join(
            common_config['log_dir'],
            'qwen.log'
        ),

    'save_path':
        os.path.join(
            common_config['data_path'],
            'ckpt/qwen.pth'
        ),

    'sid_method': rq_ave['sid_method'],
    'model_name':'Qwen',

}