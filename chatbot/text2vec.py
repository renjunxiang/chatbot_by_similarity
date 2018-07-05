from gensim.models import word2vec
import numpy as np


def creat_dict(texts_cut=None,
               sg=1,
               size=128,
               window=5,
               min_count=1):
    '''
    训练词向量模型词典
    :param texts_cut: Word list of texts
    :param sg: 0 CBOW,1 skip-gram
    :param size: The dimensionality of the feature vectors
    :param window: The maximum distance between the current and predicted word within a sentence
    :param min_count: Ignore all words with total frequency lower than this
    :return:
    '''
    model_word2vec = word2vec.Word2Vec(texts_cut, sg=sg, size=size, window=window, min_count=min_count)
    return model_word2vec


def text2vec(texts_cut=None,
             model_word2vec=None,
             merge=True):
    '''
    文本的词语序列转为词向量序列
    :param texts_cut: Word list of texts
    :param model_word2vec: word2vec model of gensim
    :param merge: If Ture, calculate sentence average vector
    :return:
    '''
    texts_vec = [[model_word2vec[word] for word in text_cut if word in model_word2vec] for text_cut in texts_cut]
    if merge:
        # 避免句子长度为0报错
        return np.array([sum(i) / max(len(i), 1.0) for i in texts_vec])
    else:
        return texts_vec


if __name__ == '__main__':
    texts_cut = [['北京', '天安门'], ['北京', '长城']]
    model_word2vec = creat_dict(texts_cut=texts_cut,
                                sg=1,
                                size=4,
                                window=5,
                                min_count=1)
    texts_vec = text2vec(texts_cut=texts_cut,
                         model_word2vec=model_word2vec,
                         merge=True)
    print('texts_vec:\n', texts_vec)
