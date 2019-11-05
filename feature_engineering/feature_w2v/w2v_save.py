'''
此脚本主要是为了解决预加载的词向量模型占用内存太大的问题。
先将它们存储为KeyedVectors形式，这样加载更快，而且内存占用小
'''
# coding: utf-8
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import numpy as np
import pandas as pd
from tqdm import tqdm

def save_gnews(vocab):
    model = KeyedVectors.load_word2vec_format('../../corpora/GoogleNews-vectors-negative300.bin', binary=True)
    # 新建KeyedVectors
    kmodel = KeyedVectors(300)
    loss = 0
    for word in vocab:
        try:
            vec = model[word]
        except:
            loss += 1
            continue
        kmodel.add(word, vec, replace=True)
    print('loss word: ', loss)
    kmodel.save('../../corpora/gnews.wv')

def save_glove(vocab):
    #model = KeyedVectors.load_word2vec_format('../../corpora/glove.840B.300d.txt', binary=False)
    kmodel = KeyedVectors(300)
    vocab = set(vocab.to_list())
    f = open('../../corpora/glove.840B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = str(values[0])
        if word not in vocab:
            continue
        try:
            vec = np.asarray(values[1:], dtype='float32')
        except:
            continue
        kmodel.add(word, vec, replace=True)
    f.close()
    kmodel.save('../../corpora/glove.wv')


def save_fasttext(vocab):
    model = FastText.load_word2vec_format('../../corpora/wiki.en.vec')
     # 新建KeyedVectors
    kmodel = KeyedVectors(300)
    loss = 0
    for word in vocab:
        try:
            vec = model[word]
        except:
            loss += 1
            continue
        kmodel.add(word, vec, replace=True)
    print('loss word: ', loss)
    kmodel.save('../../corpora/fasttext.wv')

if __name__ == '__main__':
    df_vocab = pd.read_csv('../../dataset/quora_vocab.csv')
    vocab = df_vocab.word
    #save_gnews(vocab)
    save_glove(vocab)
    #save_fasttext(vocab)
