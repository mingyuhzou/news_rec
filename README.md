# Two-Stage News Recommendation System

本项目实现一个基于 MIND 的两阶段新闻推荐系统—— **召回（Recall）+ 排序（Rank）** 架构，在保证效率的同时提升推荐精度。

数据集使用Mind，来自https://msnews.github.io/，是微软亚洲研究院基于 Microsoft News 平台的匿名用户行为日志整理而成，包含约10万条新闻内容以及超过260万条用户浏览记录，覆盖约75万名用户的点击与未点击行为。这里用的是阉割版本，因为完整版本不开放了。

数据表如下

news.csv数据示例

| 列名               | 内容                                                         |
| ------------------ | ------------------------------------------------------------ |
| news_ id           | N1                                                           |
| category           | sports                                                       |
| subcategory  title | football_nfl  Texans defensive tackle D.J. Reader is taking advantage of  his opportunities |
| abstract           | Houston Texans defensive tackle D.J. Reader is taking  advantage of opportunities given by defensive end J.J. Watt |

behaviors.csv数据示例

| 列名          | 内容                                                   |
| ------------- | ------------------------------------------------------ |
| id            | 1                                                      |
| user_id       | U87243                                                 |
| time  history | 2019-11-10  11:30:54  ['N8668', 'N39081', …, 'N64932'] |
| impressions   | ['N78206-0', 'N26368-1', …, 'N27822-0']                |

下载得到的数据格式有问题，需手动处理。



召回模型参考https://github.com/XiaoLongtaoo/TIGER，实现Tiger，做生成式召回，该模型比较适合Mind的数据（缺少用户特征，有完整的历史点击序列）。Tiger所需的物品embedding通过在本地部署nlp模型得到，可以使用all-MiniLM-L6-v2。

排序模型复现19年的NRMS，Neural News Recommendation with Multi-Head Self-Attention，该模型基于Mind数据集实现，结构相对简单，易于实现。



最终指标如下

recall: 100%|██████████| 484/484 [00:42<00:00, 11.33it/s]
{'Recall@5': 0.04923911317066897, 'Recall@10': 0.08182347969590756, 'Recall@15': 0.1042179884504496, 'Recall@20': 0.1207220809734311} {'NDCG@5': 0.03160378140154918, 'NDCG@10': 0.042046421689023686, 'NDCG@15': 0.047971191176228715, 'NDCG@20': 0.05186834514202658}

送入到排序后，得到

rank: 100%|██████████| 46420/46420 [00:19<00:00, 2417.05it/s]
{'NDCG@5': np.float64(0.1805528796603235), 'NDCG@10': np.float64(0.24445171601196938), 'NDCG@15': np.float64(0.29042336937824365), 'NDCG@20': np.float64(0.32589752184735105), 'AUC': np.float64(0.6214888219765714)}



