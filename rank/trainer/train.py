import torch
import torch.nn as nn
from config.model.nrms import hparams
from rank.model.NRMS import NRMS
import pytorch_optimizer as optim_
from rank.data.dataset import trainDataset
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from rank.utils.processData import processW2vec

processW2vec(hparams['news_file'], hparams['vocab_file'], hparams['w2v_file'])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
news_file=hparams['news_file']
behaviors_file=hparams['behaviors_file']
w2v_file=hparams['w2v_file']

# 取数据
train_dataset=trainDataset(news_file,behaviors_file,w2v_file,max_len=20,max_hist_len=50,neg_num=4)
train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)

# 打开预训练词嵌入
with open(w2v_file, 'rb') as f:
    data = pickle.load(f)
    w2id = data['w2id']
    matrix = torch.from_numpy(data['embedding']).float() # 这就是从 GloVe 提取的矩阵
hparams['vocab_size'] = matrix.shape[0]

model=NRMS(hparams,matrix)
model=model.to(device)
criterion=nn.CrossEntropyLoss()

# optimizer=optim.Adam(model.parameters(),lr=hparams['lr'])
# optimizer=optim.Adam(model.parameters(),lr=hparams['lr'],weight_decay=1e-5)

# 效果更好
optimizer=optim_.Ranger(model.parameters(),lr=hparams['lr'])

# 缩放，防止梯度下溢
scaler = torch.amp.GradScaler('cuda')

def train():
    model.train()
    for epoch in range(1,hparams['epochs']+1):
        total_loss=0
        for click_docs,cand_docs,labels in tqdm(train_loader):
            click_docs=click_docs.to(device)
            cand_docs=cand_docs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            # 开启自动混合精度上下文，让GPU自动使用更低的精度，减少显存加速计算
            with torch.amp.autocast('cuda'):
                #计算式自动选择合适的精度
                scores=model(click_docs,cand_docs)
                loss=criterion(scores,labels)

            # 使用scaler缩放精度并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # 更新缩放因子
            scaler.update()

            total_loss+=loss.item()
        if epoch==1 or epoch%5==0:print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

train()

model_path=hparams['model_path']
torch.save(model.state_dict(), model_path)