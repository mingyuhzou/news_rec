import os

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from data.rqave_dataset import EmbDataset
from config.model.rq_ave import cfg as model_cfg
from config.process.generate_code import cfg as process_cfg
from recall.rq_ave.model.RQ_VAE import RQVAE
from recall.utils.collision import check_collision, get_collision_item, get_indices_count

def main():
    # 1. 数据集和 DataLoader
    data =EmbDataset(os.path.join(model_cfg['embed_path'],'item_emb_title.parquet'))

    dataloader = DataLoader(
        data,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=model_cfg["num_workers"]
    )

    # 2. 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. 模型和 checkpoint
    model = RQVAE(
            model_cfg["in_dim"],
            model_cfg["num_emb_list"],
            model_cfg["e_dim"],
            model_cfg["layers"],
            model_cfg["dropout_prob"],
            model_cfg["bn"],
            model_cfg["loss_type"],
            model_cfg["quant_loss_weight"],
            model_cfg["beta"],
            model_cfg["kmeans_init"],
            model_cfg["kmeans_iters"],
            model_cfg["sk_epsilons"],
            model_cfg["sk_iters"],
    )

    ckpt_path =process_cfg['model_weight_path']
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    # 4. 保存结果的容器
    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    # 生成离散码，初始时不用sk，因为会导致大量重复，先生成随机，后面再用sk解决冲突
    for d in tqdm(dataloader):
        d = d.to(device)  # [B, in_dim] ([1024, 768])
        indices = model.get_indices(d, use_sk=False)  # [B, L(层数)] ([1024, 3])
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()  # [B, L] (1024, 3)
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices) # (104151(新闻数), 3)
    all_indices_str = np.array(all_indices_str) # ["['<a_226>', '<b_248>', '<c_1>']" "['<a_180>', '<b_11>', '<c_248>']""['<a_15>', '<b_219>', '<c_40>']" "['<a_63>', '<b_199>', '<c_90>']""['<a_202>', '<b_229>', '<c_243>']"]

    # 6. 关闭前几层 Sinkhorn（硬量化）即前几层保持软编码减少初始重复
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0

    tt=0
    # 对重复离散码使用硬编码
    while True:
        if tt>=30 or check_collision(all_indices_str):
            break
        collision_item_groups=get_collision_item(all_indices_str)
        for collision_items in collision_item_groups:
            d=data[collision_items].to(device)

            indices=model.get_indices(d,use_sk=True)
            indices=indices.view(-1,indices.shape[-1]).cpu().numpy()
            for item,index in zip(collision_items,indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                all_indices[item]=code
                all_indices_str[item]=str(code)
        tt+=1
    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict={}
    for item ,indices in enumerate(all_indices.tolist()):
        all_indices_dict[item]=indices

    codes=[]

    # 简化编码格式
    for key,value in all_indices_dict.items():
        code=[int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes)
    # 构造新的维度
    codes_array=np.hstack((codes_array,np.zeros((codes_array.shape[0],1),dtype=int)))

    unique_codes,counts=np.unique(codes_array,axis=0,return_counts=True)
    duplicates=unique_codes[counts>1]

    # 对重复编码在最后一个维度上增加
    if len(duplicates)>0:
        print("Resolving duplicates in codes...")
        for duplicate in duplicates:
            duplicate_indices=np.where((codes_array==duplicate).all(axis=1))[0]
            for i,idx in enumerate(duplicate_indices):
                codes_array[idx,-1]=i
    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]

    if len(duplicates) > 0:
        print("There still have duplicates:", duplicates)
    else:
        print("There are no duplicates in the codes after resolution.")
    print(f"Saving codes to {process_cfg['output_file']}")
    print(f"the first 5 codes: {codes_array[:5]}")
    np.save(process_cfg['output_file'], codes_array)

if __name__ == "__main__":
    main()