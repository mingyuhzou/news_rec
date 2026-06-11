# Two-Stage News Recommendation System

本项目实现了一个基于 [MIND](https://msnews.github.io/) 数据集的两阶段新闻推荐系统，整体采用 **Recall + Rank** 架构，在控制计算成本的同时提升推荐效果。

项目可在 ** ubuntu 25.10 5070ti 16GB 显存 python 3.10*环境下可运行。

## 项目简介

本项目包含两个核心阶段：

1. **召回阶段（Recall）**
   参考 [TIGER](https://github.com/XiaoLongtaoo/TIGER) 实现生成式召回模型。
   该方法较适合 MIND 数据集，因为 MIND 缺少显式用户特征，但提供了完整的用户历史点击序列。模型通过用户历史行为生成候选新闻的 Semantic ID，从而完成候选集召回。
2. **排序阶段（Rank）**
   复现 2019 年的 [NRMS](https://aclanthology.org/D19-1671/) 模型, *Neural News Recommendation with Multi-Head Self-Attention*。
   NRMS 使用多头自注意力机制建模新闻表示和用户表示，结构清晰，易于实现，适合作为新闻推荐排序模型。

## 数据集

本项目使用 **MIND small** 数据集。

MIND 是微软亚洲研究院基于 Microsoft News 平台匿名用户行为日志构建的新闻推荐数据集，包含新闻内容、用户历史点击行为以及曝光日志。本文使用的是 `MIND small` 版本。

### news.csv 示例

| 列名        | 示例                                                         |
| ----------- | ------------------------------------------------------------ |
| news_id     | N1                                                           |
| category    | sports                                                       |
| subcategory | football_nfl                                                 |
| title       | Texans defensive tackle D.J. Reader is taking advantage of his opportunities |
| abstract    | Houston Texans defensive tackle D.J. Reader is taking advantage of opportunities given by defensive end J.J. Watt |

### behaviors.csv 示例

| 列名        | 示例                                        |
| ----------- | ------------------------------------------- |
| id          | 1                                           |
| user_id     | U87243                                      |
| time        | 2019-11-10 11:30:54                         |
| history     | `['N8668', 'N39081', ..., 'N64932']`        |
| impressions | `['N78206-0', 'N26368-1', ..., 'N27822-0']` |



### Recall：TIGER-style 生成式召回

召回模型参考 [TIGER](https://github.com/XiaoLongtaoo/TIGER)实现。

TIGER 的核心思想是先将物品表示离散化为 Semantic ID，然后将推荐任务转化为序列生成任务(history click -> next click)。对于新闻推荐场景，可以使用新闻标题或正文生成新闻 embedding，再通过 RQ-VAE 等方法得到离散 Semantic ID，最后训练 Transformer 模型根据用户历史点击序列生成目标新闻 ID。

本项目中，新闻 embedding 通过本地部署文本编码模型生成，默认使用：

- [all-MiniLM-L6-v2](https://modelscope.cn/models/sentence-transformers/all-MiniLM-L6-v2)

### Rank：NRMS 排序模型

排序模型复现 NRMS，即 *Neural News Recommendation with Multi-Head Self-Attention*。

NRMS 主要包含：

- 新闻编码器：使用多头自注意力建模新闻标题中的词级语义信息；
- 用户编码器：使用多头自注意力建模用户历史点击新闻之间的关系；
- 点击预测层：计算用户表示与候选新闻表示之间的匹配分数。

## 环境要求

建议环境：

```text
Python >= 3.10
PyTorch >= 1.12
CUDA >= 11.0
GPU memory >= 16GB
```

安装依赖：

```bash
pip install modelscope
```

如项目中提供了 `requirements.txt`，也可以使用：

```bash
pip install -r requirements.txt.txt
```

## 数据准备

### 1. 下载 MIND small 数据集

从 [MIND 官网](https://msnews.github.io/) 下载 zip 格式数据集：`MINDsmall_train.zip`

将数据集放入 `Data` 目录下，例如：

```text
Data/
├── MINDsmall_train.zip
```

### 2. 下载文本编码模型

下载 [all-MiniLM-L6-v2](https://modelscope.cn/models/sentence-transformers/all-MiniLM-L6-v2) 到 `Data` 目录下：

```bash
modelscope download \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --local_dir Data/all-MiniLM-L6-v2
```



## 使用方法

### 1. 数据预处理与新闻 embedding 生成

运行 notebook：

```text
notebooks/embedding.ipynb
```

- 解析 MIND 原始数据；
- 生成 `news.csv` 和 `behaviors.csv`；
- 划分训练集、验证集和测试集；
- 使用 `all-MiniLM-L6-v2` 根据新闻标题生成新闻 embedding。

### 2. 生成 Semantic ID
从DATA/ckpt 选择RQ—VAE权重，填写到config/generate的model_weight_path 

运行 RQ-VAE 模块，将新闻 embedding 离散化为 Semantic ID：

```bash
python recall/rq-vae/main.py
```

该步骤输出每条新闻对应的离散 Semantic ID，供生成式召回模型训练使用。

### 3. 训练召回模型

运行 Transformer 召回模型：

```bash
python recall/transformer/main.py
```

该阶段根据用户历史点击序列训练生成式召回模型，并在验证集上评估召回效果。

### 4. 训练排序模型

运行 NRMS 排序模型，对召回阶段得到的候选新闻进行精排。

如果项目中已提供排序入口，可按实际路径执行，例如：

```bash
python rank/nrms/main.py
```

## 训练耗时

在 16GB 显存环境下，召回模型训练耗时约为：

| 阶段            | 时间      |
| --------------- | --------- |
| 单个 epoch 训练 | 约 40 秒  |
| 单次验证        | 约 2 分钟 |

实际耗时会受到 GPU 型号、batch size、序列长度和候选集规模影响。

## 项目目录

```text
.
├── Data/
│   ├── MINDsmall_train/
│   └── all-MiniLM-L6-v2/
│
├── notebooks/
│   └── embedding.ipynb
│
├── recall/
│   ├── rq-vae/
│   │   └── main.py
│   └── transformer/
│       └── main.py
│
├── rank/
│   └── nrms/
│       └── main.py
│
├── README.md
└── requirements.txt
```

## 指标
Tiger指标

|  指标   |          值          |
| :-----: | :------------------: |
|  Hit@1  | 0.007167844085221587 |
| Hit@10  | 0.023023105741813484 |
| Hit@20  | 0.042645659535136736 |
| NDCG@1  | 0.007167844085221587 |
| NDCG@10 | 0.013686004393870238 |
| NDCG@20 | 0.018574634632242938 |
