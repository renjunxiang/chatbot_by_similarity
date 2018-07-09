from . import cut_texts, creat_dict, text2vec, cal_similarity
import pandas as pd
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')


class chatbot():
    def __init__(self):
        self.texts_all = []
        self.model_word2vec = None
        self.texts_vec = None

    def train(self, texts, reuse=True, mode='knowledge'):
        '''
        训练语料库得到句向量
        :param texts: list,语料库列表
        :param reuse: bool,是否增加语料库,True添加,False覆盖
        :param mode: str,回答的类型,’knowledge' or 'chat'
        :return:
        '''
        # 聊天模式词语全留,知识库模式词语长度2
        if mode == 'chat':
            word_len = 1
        elif mode == 'knowledge':
            word_len = 2
        else:
            raise ValueError('mode should be knowledge or chat')
        self.mode = mode
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
                              word_len=word_len)
        print('finish cut texts')

        # 训练词向量
        model_word2vec = creat_dict(texts_cut=texts_cut,
                                    sg=0,
                                    size=128,
                                    window=5,
                                    min_count=1)
        self.model_word2vec = model_word2vec
        print('finish word2vec')

        # 计算句向量
        texts_vec = text2vec(texts_cut=texts_cut,
                             model_word2vec=model_word2vec,
                             merge=True)
        self.texts_vec = texts_vec
        print('finish text2vec')

    def get_answer(self,
                   ask='',
                   sample=50000,
                   similarity='cos',
                   modify=False,
                   threshold=0.0,
                   topn=2):
        '''
        根据问题找到相似内容
        :param ask: str,问题
        :param sample: int,答案抽样数量,避免计算过慢
        :param similarity: str,'cos' or 'Euclidean',相似度计算方法
        :param modify: bool,是否进行余弦修正
        :param threshold: float,相似度阈值
        :param topn: int,返回的答案数
        :return:
        '''
        mode = self.mode
        if mode == 'chat':
            threshold = -2
            word_len = 1
        elif mode == 'knowledge':
            word_len = 2
        else:
            raise ValueError('mode should be knowledge or chat')
        texts_vec = self.texts_vec
        texts_all = self.texts_all
        if len(texts_vec)>sample:
            start_index=random.sample(range(len(texts_vec)-sample),1)[0]
            texts_vec=texts_vec[start_index:start_index+sample]
            texts_all=texts_all[start_index:start_index+sample]
        # 对问题分词
        ask_cut = cut_texts(texts=[ask],
                            need_cut=True,
                            word_len=word_len)

        # 计算问题的向量
        ask_vec = text2vec(texts_cut=ask_cut,
                           model_word2vec=self.model_word2vec,
                           merge=True)

        # 计算问题和知识库每句话的余弦值,部分不能计算的相似度记为-999
        similarity_all = []
        if modify:
            texts_vec_use = np.array([i for i in texts_vec if type(i) == np.ndarray])
            texts_vec_mean = texts_vec_use.mean(axis=0)
            for i in texts_vec:
                # 出现nan则相似度记为-999
                if type(ask_vec[0]) != np.ndarray or type(i) != np.ndarray:
                    similarity_one = -999.0
                else:
                    similarity_one = cal_similarity(v1=ask_vec[0] - texts_vec_mean,
                                                    v2=i - texts_vec_mean,
                                                    similarity=similarity)
                similarity_all.append(similarity_one)
        else:
            for i in texts_vec:
                # 出现nan则相似度记为-999
                if type(ask_vec[0]) != np.ndarray or type(i) != np.ndarray:
                    similarity_one = -999.0
                else:
                    similarity_one = cal_similarity(v1=ask_vec[0], v2=i, similarity=similarity)
                similarity_all.append(similarity_one)

        texts_index = np.arange(len(texts_all))
        text_similarity = pd.DataFrame({'texts_index': texts_index,
                                        'similarity': similarity_all},
                                       columns=['texts_index', 'similarity'])
        # 按余弦值从高到低排序
        text_similarity_sort = text_similarity.sort_values(by='similarity', ascending=False)

        # 筛选出余弦值大于阈值，前几个答案
        ask_samilarity_index = list(text_similarity_sort.loc[text_similarity_sort['similarity'] >= threshold,
                                                             'texts_index'])[:topn]

        if mode == 'knowledge':
            if ask_samilarity_index == []:
                print('没有找到匹配的内容')
            else:
                print('按照匹配得分从高到低，您的问题 “%s” 和知识库的这些内容相关：\n' % (ask))
                for n, i in enumerate(ask_samilarity_index):
                    print('知识%d: %s' % (n + 1, texts_all[i]))
        elif mode == 'chat':
            if ask_samilarity_index == []:
                print('不明白你在说什么==！')
            else:
                ask_samilarity_index_random = random.sample(ask_samilarity_index, 1)[0]
                # 避免抽中最后一个
                while ask_samilarity_index_random == len(texts_all):
                    ask_samilarity_index_random = random.sample(ask_samilarity_index, 1)[0]
                print(texts_all[ask_samilarity_index_random + 1])
        else:
            pass
