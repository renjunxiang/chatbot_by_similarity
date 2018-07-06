import numpy as np


def cal_similarity(v1, v2, mode='cos'):
    """
    计算余弦相似度
    :param v1: 第一个向量
    :param v2: 第二个向量
    :param modify: bool,是否进行余弦修正
    :return: 余弦相似度
    """

    if mode == 'cos':
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cos
    elif mode == 'Euclidean':
        Euclidean = np.linalg.norm(v2 - v1)
        return Euclidean
    else:
        raise ValueError('mode should be cos or Euclidean')


if __name__ == '__main__':
    x = np.array([1, 2])
    y = np.array([1, 1])
    print('cos:', cal_similarity(x, y, mode='cos'))
    print('Euclidean:', cal_similarity(x, y, mode='Euclidean'))
