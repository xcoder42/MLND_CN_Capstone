### Quora句子相似度匹配

Quora 数据集训练集共包含40K的句子对，且其完全来自于Quora网站自身，Quora在发布数据集的同时，在Kaggle平台，发起了Quora句子相似度匹配大赛，共有3307支队伍参加了本次句子相似度匹配大赛，参赛队伍不仅包括来自麻省理工学院、伦敦大学学院、北京大学、清华大学、中科院计算所等高校研究所，也包括了来自微软、Airbnb、IBM等工业界的人员。

---
### 项目运行环境
* 操作系统：win10
* cpu：i7-8750H
* gpu：NVIDIA GTX1060
---
### 项目使用的库
* numpy
* pandas
* matplotlib
* keras-gpu
* tensorflow-gpu
* fuzzywuzzy
* scikit-learn
* scipy
* lightgbm
* nltk
* gensim
* networkx
---
### 数据资源
* stanford glove词向量：http://www-nlp.stanford.edu/data/glove.840B.300d.zip
* google news 词向量：https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

---
### 目录及文件说明
* dataset: 存放基础数据集
* corpora：存放词向量文件
* feature_engineering: 特征工程代码
* feature_store: 特征文件
* model_bulid: 深度类模型构建代码
* model_store: 模型保存
* submission：结果提交文件
* data_explore.ipynb: 数据探索notebook
* dnn_model.ipynb: 深度类模型实验notebook
* lgbm_model.ipynb: lightGBM模型实验notebook
* ensemble_model.ipynb: 模型融合实验notebook
* evaluate.ipynb: 结果评估notebook




