from . import cut_texts, creat_dict, text2vec, cal_similarity
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class chatbot():
    def __init__(self):
        self.texts_all = []
        self.model_word2vec = None
        self.texts_vec = None

    def train(self, texts, reuse=True):
        '''
        训练语料库得到句向量
        :param texts: list,语料库列表
        :param reuse: bool,是否增加语料库,True添加,False覆盖
        :return:
        '''
        # 添加进原有知识列表
        if reuse:
            texts_all = self.texts_all
            texts_all += texts
        else:
            texts_all = texts
        self.texts_all = texts_all
        # 分词
        texts_cut = cut_texts(texts=texts_all,
                              need_cut=True,
                              word_len=2)

        # 训练词向量
        model_word2vec = creat_dict(texts_cut=texts_cut,
                                    sg=0,
                                    size=128,
                                    window=5,
                                    min_count=1)
        self.model_word2vec = model_word2vec

        # 计算句向量
        texts_vec = text2vec(texts_cut=texts_cut,
                             model_word2vec=model_word2vec,
                             merge=True)
        self.texts_vec = texts_vec

    def get_answer(self,
                   ask='',
                   mode='cos',
                   modify=False,
                   threshold=0.0,
                   topn=2):
        '''
        根据问题找到相似内容
        :param ask: str,问题
        :param mode: str,'cos' or 'Euclidean',相似度计算方法
        :param modify: bool,是否进行余弦修正
        :param threshold: float,相似度阈值
        :param topn: int,返回的答案数
        :return:
        '''
        texts_vec = self.texts_vec
        # 对问题分词
        ask_cut = cut_texts(texts=[ask],
                            need_cut=True,
                            word_len=2)

        # 计算问题的向量
        ask_vec = text2vec(texts_cut=ask_cut,
                           model_word2vec=self.model_word2vec,
                           merge=True)

        # 计算问题和知识库每句话的余弦值,部分不能计算的余弦值记为-999
        similarity_all = []
        if modify:
            texts_vec_use = np.array([i for i in texts_vec if type(i) == np.ndarray])
            texts_vec_mean = texts_vec_use.mean(axis=0)
            for i in texts_vec:
                # 出现空向量则相似度记为-999
                if type(ask_vec[0]) != np.ndarray or type(i) != np.ndarray:
                    similarity_one = -999.0
                else:
                    similarity_one = cal_similarity(v1=ask_vec[0] - texts_vec_mean, v2=i - texts_vec_mean)
                similarity_all.append(similarity_one)
        else:
            for i in texts_vec:
                # 出现空向量则相似度记为-999
                if type(ask_vec[0]) != np.ndarray or type(i) != np.ndarray:
                    similarity_one = -999.0
                else:
                    similarity_one = cal_similarity(v1=ask_vec[0], v2=i)
                similarity_all.append(similarity_one)

        text_similarity = pd.DataFrame({'text': self.texts_all,
                                        'similarity': similarity_all},
                                       columns=['text', 'similarity'])
        # 按余弦值从高到低排序
        text_similarity_sort = text_similarity.sort_values(by='similarity', ascending=False)

        # 筛选出余弦值大于阈值，前几个答案
        ask_same = list(text_similarity_sort.loc[text_similarity_sort['similarity'] >= threshold, 'text'])[:topn]
        if ask_same == []:
            print('没有找到匹配的内容')
        else:
            print('按照匹配得分从高到低，您的问题 “%s” 和知识库的这些内容相关：\n' % (ask))
            for n, i in enumerate(ask_same):
                print('知识%d: %s' % (n + 1, i))
