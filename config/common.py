from os.path import join
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
data_path = ROOT_DIR / "tmp"

cfg={
    "data_path":data_path,
    "train_data_path": join(data_path,'train'),
    "test_data_path": join(data_path,'test'),
    "emb_model_path":join(data_path,'all-MiniLM-L6-v2'),
    "log_dir":join(ROOT_DIR,"log"),
    "embed_path":join(data_path,'embedding'),
    "user_dict":join(data_path,'user_dict.npy'),
    "item_dict":join(data_path,'item_dict.npy'),
    'news_file':os.path.join(data_path,'news.parquet'),

    'code2item':os.path.join(data_path,'code2item.npy'),

    'eval_data':os.path.join(data_path,'dev','test_df.parquet'),
    'metric':os.path.join(data_path,'dev','metric.parquet'),
}
