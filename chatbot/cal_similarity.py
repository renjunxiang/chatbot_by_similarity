import numpy as np


def cal_cos(v1, v2):
    """
    计算tf-idf的余弦相似度
    :param v1: 第一个向量
    :param v2: 第二个向量
    :return: 余弦相似度
    """
    if type(v1) != np.ndarray or type(v2) != np.ndarray:
        cos = -999.0
    else:
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos


if __name__ == '__main__':
    x = np.array([1, 1])
    y = np.array([1, 2])
    print('cos:',cal_cos(x, y))
