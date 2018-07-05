import jieba

jieba.setLogLevel('WARN')


def cut_texts(texts=None, need_cut=True, word_len=1):
    '''
    Use jieba to cut texts
    :param texts:list of texts
    :param need_cut:whether need cut text
    :param word_len:min length of words to keep,in order to delete stop-words
    :param savepath:path to save word list in json file
    :return:
    '''
    if need_cut:
        if word_len > 1:
            texts_cut = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in texts]
        else:
            texts_cut = [jieba.lcut(one_text) for one_text in texts]
    else:
        if word_len > 1:
            texts_cut = [[word for word in text if len(word) >= word_len] for text in texts]
        else:
            texts_cut = texts

    return texts_cut


if __name__ == '__main__':
    texts = ['我爱北京天安门', '我爱北京长城']
    print('results:', cut_texts(texts=texts, need_cut=True, word_len=2))
