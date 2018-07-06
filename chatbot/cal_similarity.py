import numpy as np


def cal_similarity(v1, v2, mode='cos', modify=False):
    """
    计算余弦相似度
    :param v1: 第一个向量
    :param v2: 第二个向量
    :param mode: str,相似度计算方法,'cos'余弦 or 'Euclidean'欧氏距离
    :param modify: bool,是否进行余弦修正
    :return: 余弦相似度
    """
    if type(v1) != np.ndarray or type(v1) != np.ndarray:
        cos = -999.0
    else:
        if mode == 'cos':
            # 考虑模长,减去向量均值
            if modify:
                v_mean = np.array([v1, v2]).mean(axis=0)
                v1 = v1 - v_mean
                v2 = v2 - v_mean
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return cos
        elif mode == 'Euclidean':
            Euclidean = np.linalg.norm(v2 - v1)
            return Euclidean
        else:
            raise ValueError('mode should be cos or Euclidean')


if __name__ == '__main__':
    x = np.array([1, 1])
    y = np.array([1, 2])
    print('cos:', cal_similarity(x, y, mode='cos'))
    print('Euclidean:', cal_similarity(x, y, mode='Euclidean'))
