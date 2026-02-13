import os
from config.common import cfg as config
from config.process.generate_code import cfg as generate_code
cfg={
    # 训练与推理通用设置
    'batch_size': 256,  # 训练时的 Batch 大小
    'infer_size': 96,  # 生成推荐结果（推理）时的 Batch 大小
    'num_epochs': 20,  # 训练轮数
    'lr': 1e-4,  # 优化器的学习率
    'device': 'cuda',  # 运行设备 (cuda 或 cpu)
    'seed': 2025,  # 随机种子，确保实验可重复性
    'early_stop': 10,  # 早停法耐心值（连续多少个 epoch 指标不提升则停止）

    # 模型架构参数 (Transformer/T5 相关)
    'num_layers': 4,  # 编码器层数
    'num_decoder_layers': 4,  # 解码器层数
    'd_model': 128,  # 隐藏层维度
    'd_ff': 1024,  # 前馈网络 (FFN) 的维度
    'num_heads': 6,  # 多头注意力机制的头数
    'd_kv': 64,  # Key 和 Value 向量的维度
    'dropout_rate': 0.1,  # Dropout 概率
    'vocab_size': 1025,  # 词表大小
    'pad_token_id': 0,  # 填充 Token 的 ID
    'eos_token_id': 0,  # 序列结束 Token 的 ID
    'feed_forward_proj': 'relu',  # 前馈层激活函数类型

    # 数据与序列处理
    'max_len': 20,  # 序列最大长度
    'dataset_path': config['train_data_path'],  # 数据集 parquet 文件路径
    'code_path':generate_code['output_file'],  # 物品编码映射文件路径
    'mode': 'train',  # 运行模式: 'train' 或 'evaluation'

    # 生成与评估
    'topk_list': [5, 10, 20, 100],  # 评估指标的 K 值列表 (Recall@K, NDCG@K)
    'beam_size': 30,  # 束搜索 (Beam Search) 的大小

    # 文件保存与日志
    'log_path': os.path.join(config['log_dir'],'tiger.log'),  # 日志保存路径
    'save_path':os.path.join(config["data_path"],"ckpt/tiger.pth"),  # 模型检查点保存路径

}