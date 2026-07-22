#  News Recommendation System


参考 [TIGER](https://github.com/XiaoLongtaoo/TIGER) 实现生成式召回模型，对比使用RQ-VAE和[RQ-KMEANS](https://github.com/EdoardoBotta/rq-kmeans)方法构造sid

该方法较适合 MIND 数据集，因为 MIND 缺少显式用户特征与物品特征，但提供了完整且丰富（mean len > 100）的用户历史点击序列。模型通过用户历史行为生成候选新闻的 Semantic ID，从而完成候选集召回。


## 数据集

本项目基于**MIND** 数据集搭建，MIND 是微软亚洲研究院基于 Microsoft News 平台匿名用户行为日志构建的新闻推荐数据集，包含新闻内容、用户历史点击行为以及曝光日志。本文使用的是 `MIND small` 版本。

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




## 环境要求

在 ubnutu 25.10 python=3.10 5070ti 下可以正常运行 

```bash
pip install -r requirements.txt.txt
```

## 数据准备

### 1. 下载 MIND small 数据集

从 [MIND 官网](https://msnews.github.io/) 下载 zip 格式数据集：`MINDsmall_train.zip`或其他版本的文件（需修改config中对应的属性）

将数据集放入 `tmp` 目录下：

```text
tmp/
├── MINDsmall_train.zip
```

### 2. 下载文本编码模型

下载 [all-MiniLM-L6-v2](https://modelscope.cn/models/sentence-transformers/all-MiniLM-L6-v2) 到 `tmp` 目录下：

```bash
modelscope download \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --local_dir tmp/all-MiniLM-L6-v2
```



## 使用方法

### 1. 数据预处理与新闻 embedding 生成

运行 preprocess,py：

```text
python preprocess.py
```

- 解析 MIND 原始数据；
- 生成 `news.csv` 和 `behaviors.csv`；
- 划分训练集、验证集和测试集；
- 使用 `all-MiniLM-L6-v2` 根据新闻标题生成新闻 embedding。

### 2. 生成 Semantic ID
从DATA/ckpt 选择RQ—VAE权重，填写到config/generate的model_weight_path，在config/process/generate_code中的sid_method可以选择编码方式（RQ-VAE/RQ-KMEANS） 

运行 RQ-VAE 模块，将新闻 embedding 离散化为 Semantic ID：

```bash
python recall/rq-vae/generate_code.py
```

该步骤输出每条新闻对应的离散 Semantic ID，供生成式召回模型训练使用。

### 3. 训练模型

运行 Transformer 召回模型：

```bash
python recall/transformer/main.py
```

该阶段根据用户历史点击序列训练生成式召回模型，并在验证集上评估召回效果。


## 训练耗时
SID构建

+ RQ-VAE 耗时`12mins`
+ RQ-KMEANS  设置早停后比RQ-VAE快，大约`3~5mins`



T5模型在给定的参数下，train的一个epoch耗时`10mins`，evaluate的一个epoch耗时`25mins`，通过调整参数可以在12GB显存下运行



## 项目目录

```text
.
├── config/
│   ├── model/
│   │     └── nrms.py # 排序模型参数
│   │     └── tf.py # 召回模型参数
│   ├── process/
│   │     └── generate_code.py # RQ-VAE &RQ-KMEANS 参数
├── Data/
│   ├── MINDsmall_train/  # 数据集
│   └── all-MiniLM-L6-v2/ # 文本编码模型
│
├── notebooks/
│   └── embedding.ipynb  # 处理数据集
│
├── recall/
│   ├── rq-vae/  # 编码sid
│   │      └── data  # dataset
│   │      └── kernels # RQ-KMEANS 算子优化
│   │      └── model   # RQ-VAE模型
│   │      └── trainer # RQ-VAE训练类
│   │      └── generate_code.py # 构造 itemid->sid，根据参数选择sid转换方法
│   └── transformer/
│   │      └── data  # dataset
│   │      └── model   # RQ-VAE模型
│   └──    └── main.py # 训练tiger模型并评估
│
├── README.md
└── requirements.txt
```

## 指标	


| 方法     | Hit@1    | Hit@10   | Hit@20   | NDCG@1   | NDCG@10  | NDCG@20  |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| RQ-VAE   | 0.007168 | 0.023023 | 0.042646 | 0.007168 | 0.013686 | 0.018575 |
| RQ-MEANS | 0.007147 | 0.025737 | 0.048941 | 0.007147 | 0.014782 | 0.020576 |

| 指标    | RQ-MEANS - RQ-VAE | 相对变化 |
| ------- | ----------------- | -------- |
| Hit@1   | -0.000021         | -0.29%   |
| Hit@10  | +0.002714         | +11.79%  |
| Hit@20  | +0.006296         | +14.76%  |
| NDCG@1  | -0.000021         | -0.29%   |
| NDCG@10 | +0.001096         | +8.01%   |
| NDCG@20 | +0.002002         | +10.78%  |
