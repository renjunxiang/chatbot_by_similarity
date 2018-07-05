# chatbot_by_similarity
根据文本相似度实现问答的聊天机器人（弱智版）

## 项目介绍
这是根据工作需求写的一个简易版本的聊天机器人，主要目的是根据问题从知识库中匹配相应的答案从而帮助使用者去更方便的查询到一些知识性内容。<br>

## 模块简介
用法比较简单，给文本列表，经过训练后去匹配问题返回相似的答案。<br>
### 模块结构
* **文本的预处理(cut_text.py)**：用于分词、剔除停用词(这里偷懒直接把长度为1的剔除)；<br>
``` python
texts = ['我爱北京天安门', '我爱北京长城']
print('results:', cut_texts(texts=texts, need_cut=True, word_len=2))

results: [['北京', '天安门'], ['北京', '长城']]
```
* **文本转向量(text2vec.py)**：通过word2vec计算词向量，然后求均值转文本向量，空闲的话我会考虑结合tf-idf计算权重优化向量加权方法；<br>
``` python
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

texts_vec:
 [[ 0.05730793  0.01469728  0.03473849  0.04489793]
 [-0.01690295  0.12111638 -0.01626978  0.02280862]]

```
* **计算相似度(cal_similarity.py)**：目前只写了最简单的余弦值，空闲的话会考虑加入余弦修正、通过监督学习计算相似度；<br>
``` python
x = np.array([1, 1])
y = np.array([1, 2])
print('cos:',cal_cos(x, y))

cos: 0.9486832980505138
```
* **聊天机器人的训练与使用(chatbot.py)**：整合前三个步骤，计算问题和知识库每一条知识的相似度，返回排名靠前的知识；<br>

### 模块用法
只需要调用chatbot.py的功能即可，具体参考demo_train.py、demo_ask&answer.py，其中训练语料是一些word文档自行准备，model_word_document.pkl是训练好的模型可以直接使用，我的知识库比较简陋；<br>
train：训练语料库，输入文本列表 texts=[xxx,xxx]<br>
get_answer：获取答案，输入 ask=问题、threshold=相似度阈值、topn返回知识数量；<br>
``` python
from chatbot import chatbot

texts = ['我爱北京天安门', '我爱北京长城']
chatbot_try = chatbot()
chatbot_try.train(texts=texts)
answer=chatbot_try.get_answer(ask=ask, threshold=0, topn=5)
```
![](https://github.com/renjunxiang/chatbot_by_similarity/blob/master/picture/chatbot.jpg)<br>