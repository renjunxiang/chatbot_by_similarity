import numpy as np


def cal_similarity(v1, v2, similarity='cos'):
    """
    计算余弦相似度
    :param v1: 第一个向量
    :param v2: 第二个向量
    :return: 余弦相似度
    """

    if similarity == 'cos':
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cos
    elif similarity == 'Euclidean':
        Euclidean = np.linalg.norm(v2 - v1)
        return Euclidean
    else:
        raise ValueError('mode should be cos or Euclidean')


def cal_similarities(v, v_list, similarity='cos', modify=False):
    '''

    :param v: array,向量
    :param v_list: iter,向量迭代器
    :param similarity: str,'cos' or 'Euclidean',相似度计算方法
    :param modify: bool,是否进行余弦修正
    :return:
    '''
    similarity_all = []
    if modify:
        texts_vec_use = np.array([i for i in v_list if type(i) == np.ndarray])
        texts_vec_mean = texts_vec_use.mean(axis=0)
        for i in v_list:
            # 出现nan则相似度记为-999
            if type(v) != np.ndarray or type(i) != np.ndarray:
                similarity_one = -999.0
            else:
                similarity_one = cal_similarity(v1=v - texts_vec_mean,
                                                v2=i - texts_vec_mean,
                                                similarity=similarity)
            similarity_all.append(similarity_one)
    else:
        for i in v_list:
            # 出现nan则相似度记为-999
            if type(v) != np.ndarray or type(i) != np.ndarray:
                similarity_one = -999.0
            else:
                similarity_one = cal_similarity(v1=v, v2=i, similarity=similarity)
            similarity_all.append(similarity_one)
    return similarity_all


if __name__ == '__main__':
    x = np.array([1, 2])
    y = np.array([1, 1])
    print('cos:', cal_similarity(x, y, similarity='cos'))
    print('Euclidean:', cal_similarity(x, y, similarity='Euclidean'))
