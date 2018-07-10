from . import cut_texts, creat_dict, text2vec, cal_similarities
import pandas as pd
import numpy as np
import random
import warnings
from multiprocessing import Process, Queue,Lock
import pickle

warnings.filterwarnings('ignore')


def mul_cal_similarities(v, v_list, v_list_index, similarity, modify, queue,lock):
    similarity_all = cal_similarities(v, v_list, similarity, modify)
    lock.acquire()
    for i in range(len(v_list)):
        queue.put([v_list_index[i],similarity_all[i]])
    lock.release()


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
                   topn=2,
                   process_num=1):
        '''
        根据问题找到相似内容
        :param ask: str,问题
        :param sample: int,答案抽样数量,避免计算过慢
        :param similarity: str,'cos' or 'Euclidean',相似度计算方法
        :param modify: bool,是否进行余弦修正
        :param threshold: float,相似度阈值
        :param topn: int,返回的答案数
        :param process_num: int,进程数
        :return:
        '''
        mode = self.mode
        texts_vec = self.texts_vec
        texts_all = self.texts_all
        if mode == 'chat':
            threshold = -2
            word_len = 1
        elif mode == 'knowledge':
            word_len = 2
        else:
            raise ValueError('mode should be knowledge or chat')

        # 对问题分词
        ask_cut = cut_texts(texts=[ask],
                            need_cut=True,
                            word_len=word_len)

        # 计算问题的向量
        ask_vec = text2vec(texts_cut=ask_cut,
                           model_word2vec=self.model_word2vec,
                           merge=True)

        # 单进程则抽样起始位置
        if process_num == 1:
            # 是否需要抽样
            if sample:
                if len(texts_vec) > sample:
                    start_index = random.sample(range(len(texts_vec) - sample), 1)[0]
                    texts_vec = texts_vec[start_index:start_index + sample]
                    texts_all = texts_all[start_index:start_index + sample]

            # 计算问题和知识库每句话的余弦值,部分不能计算的相似度记为-999
            similarity_all = cal_similarities(v=ask_vec[0],
                                              v_list=texts_vec,
                                              similarity=similarity,
                                              modify=modify)
            texts_index = np.arange(len(texts_all))
            text_similarity = pd.DataFrame({'texts_index': texts_index,
                                            'similarity': similarity_all},
                                           columns=['texts_index', 'similarity'])
        # 多进程分块计算相似度
        elif process_num > 1:
            index_start = 0
            texts_vec_cut = []
            index_all_cut = []
            # 数据分块
            for i in range(process_num):
                texts_vec_cut.append(texts_vec[index_start:(index_start + sample)])
                len_n = len(texts_vec[index_start:(index_start + sample)])
                index_all_cut.append(range(index_start, index_start + min(len_n, sample)))
                index_start += index_start + sample
                if index_start > len(texts_vec):
                    process_num = i + 1
                    break

            # 创建子进程
            # print('start multiprocessing')
            queue = Queue()#超过800卡死
            lock=Lock()
            process_list = []
            for i in range(process_num):
                po = Process(target=mul_cal_similarities,
                             kwargs={'v': ask_vec[0],
                                     'v_list': texts_vec_cut[i],
                                     'v_list_index': index_all_cut[i],
                                     'similarity': similarity,
                                     'modify': modify,
                                     'queue': queue,
                                     'lock': lock})
                process_list.append(po)
            # 启动子进程
            for process in process_list:
                process.start()
            # 等待子进程全部结束
            for process in process_list:
                process.join()
            # print('finish multiprocessing')

            similarity_all=[]
            while 1:
                try:
                    similarity_one = queue.get_nowait()
                    similarity_all.append(similarity_one)
                except:
                    break
            text_similarity = pd.DataFrame(similarity_all,
                                           columns=['texts_index', 'similarity'])

        else:
            raise ValueError('process_num should be int and >=1')

        # 按余弦值从高到低排序
        text_similarity_sort = text_similarity.sort_values(by='similarity', ascending=False)

        # 筛选出余弦值大于阈值，前几个答案
        ask_samilarity_index = list(text_similarity_sort.loc[text_similarity_sort['similarity'] >= threshold,
                                                             'texts_index'])[:topn]

        if mode == 'knowledge':
            if ask_samilarity_index == []:
                return ['没有找到匹配的内容']
            else:
                return [texts_all[i] for i in ask_samilarity_index]
        elif mode == 'chat':
            if ask_samilarity_index == []:
                return '不明白你在说什么==！'
            else:
                ask_samilarity_index_random = random.sample(ask_samilarity_index, 1)[0]
                # 避免抽中最后一个
                while ask_samilarity_index_random == len(texts_all):
                    ask_samilarity_index_random = random.sample(ask_samilarity_index, 1)[0]
                return texts_all[ask_samilarity_index_random + 1]
        else:
            pass
