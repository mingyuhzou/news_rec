from os.path import join
import os

data_path='/home/ming/news_rec/Data'
cfg={
    "data_path":data_path,
    "train_data_path": join(data_path,'train'),
    "dev_data_path": join(data_path,'dev'),
    "emb_model_path":'../Embedding_model',
    "log_dir":"/home/ming/news_rec/log",
    "embed_path":join(data_path,'embedding'),
    "user_dict":join(data_path,'user_dict.npy'),
    "item_dict":join(data_path,'item_dict.npy'),
    'news_file':os.path.join(data_path,'news.parquet'),
}
