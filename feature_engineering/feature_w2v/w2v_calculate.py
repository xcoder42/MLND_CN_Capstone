import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis

stop_words = stopwords.words('english')

def word_mover_distance(model, s1, s2):
    '''
    词移距离：计算两个文档间的相似性
    在词向量的基础上，计算文档A中的词"转换"到文档B中的词的最小距离，以此作为相似度的度量
    '''
    s1 = word_tokenize(str(s1).lower())
    s2 = word_tokenize(str(s2).lower())
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

def sentence2vec(model, s):
    '''
    将句子转换为向量表示
    将词向量加起来，再做一个归一化处理
    '''
    words = str(s).lower()
    words = word_tokenize(words)
    # 取不属于停用词和是英文的word
    words = [w for w in words if (not w in stop_words) and (w.isalpha())]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    v2_sum = (v ** 2).sum()
    if v2_sum <= 0:
        return np.zeros(300)
    else:
        return v / np.sqrt(v2_sum)

def zip_vec(v1, v2):
    return zip(np.nan_to_num(v1), np.nan_to_num(v2))

def cosine_distance(v1, v2):
    '''
    计算句向量的余弦距离
    '''
    return [cosine(x, y) for (x, y) in zip_vec(v1, v2)]

def cityblock_distance(v1, v2):
    '''
    计算曼哈顿距离：|x1 - x2| + |y1 - y2|
    '''
    return [cityblock(x, y) for (x, y) in zip_vec(v1, v2)]

def jaccard_distance(v1, v2):
    '''
    计算杰卡德距离（1 - 杰卡德相似系数）：1 - （交集个数 / 并集个数）
    '''
    return [jaccard(x, y) for (x, y) in zip_vec(v1, v2)]

def canberra_distance(v1, v2):
    '''
    '''
    return [canberra(x, y) for (x, y) in zip_vec(v1, v2)]

def euclidean_distance(v1, v2):
    '''
    '''
    return [euclidean(x, y) for (x, y) in zip_vec(v1, v2)]

def minkowski_distance(v1, v2):
    '''
    '''
    return [minkowski(x, y) for (x, y) in zip_vec(v1, v2)]

def braycurtis_distance(v1, v2):
    '''
    '''
    return [braycurtis(x, y) for (x, y) in zip_vec(v1, v2)]

def vec_skew(v):
    '''
    计算数据集的偏度
    偏度(Skewness)亦称偏态、偏态系数
    '''
    return [skew(x) for x in np.nan_to_num(v)]

def vec_kurtosis(v):
    '''
    计算峰度系数
    表征概率密度分布曲线在平均值处峰值高低的特征数
    '''
    return [kurtosis(x) for x in np.nan_to_num(v)]