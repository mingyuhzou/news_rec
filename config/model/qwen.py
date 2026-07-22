import os

from config.common import cfg as common_config
from config.process.generate_code import cfg as generate_code
from config.model.generate_sid import cfg as rq_ave


cfg={

    # =====================
    # training
    # =====================

    'batch_size':128,
    'infer_size':96,
    'num_epochs':51,
    'lr':1e-4,
    'device':'cuda',
    'seed':2025,
    'early_stop':10,
    # =====================
    # Qwen architecture
    # =====================

    # vocab
    'vocab_size':1025,
    'pad_token_id':0,
    'eos_token_id':1,
    # hidden dimension
    # 对应T5 d_model
    'hidden_size':128,
    # 对应T5 d_ff
    'intermediate_size':512,
    # Transformer层数
    # 对应T5 num_layers
    'num_hidden_layers':4,
    # attention head
    # 对应T5 num_heads
    'num_attention_heads':4,

    # 最大输入长度
    #
    # Qwen:
    # history SID + target SID
    #
    'max_position_embeddings':2048,


    # =====================
    # data
    # =====================

    'max_len':100,


    'dataset_path':
        common_config['data_path'],


    'code_path':
        generate_code['output_file'],


    'mode':'train',


    # SID长度
    #
    # Qwen生成:
    # code1 code2 code3 EOS
    #
    'code_len':
        len(rq_ave['num_emb_list'])+1,



    # =====================
    # generation
    # =====================

    'beam_size':25,
    # =====================
    # save
    # =====================

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

}