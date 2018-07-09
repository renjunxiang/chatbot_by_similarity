import re
import os

localpath = os.path.dirname(__file__)
def load_data():
    f = open(localpath+'/data/xiaohuangji50w_nofenci.conv', encoding='utf-8')
    data = []
    line = True
    while line:
        line = f.readline()
        if line != 'E\n':
            re_func = re.compile('[M ,\n]')
            data.append(re_func.sub('', line))
    f.close()
    return data
