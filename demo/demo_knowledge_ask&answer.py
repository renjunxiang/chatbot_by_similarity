import pickle
import warnings

warnings.filterwarnings('ignore')

bye = '很高兴为您服务，再见！'

with open('./model_word_document.pkl', mode='rb') as f:
    chatbot_try = pickle.load(f)

print('我是机器人Wonder，很高兴见到你！\n\n')
state = True
while state:
    print('请输入您的问题（范例：谁负责全国医疗器械监督管理工作？）\n离开请输入\"quit\"：')
    ask = input()
    if ask == 'quit':
        print(bye)
        break
    else:
        answer = chatbot_try.get_answer(ask=ask,
                                        similarity='cos',
                                        modify=False,
                                        threshold=0,
                                        topn=3)
        print('按照匹配得分从高到低，您的问题 “%s” 和知识库的这些内容相关：\n' % (ask))
        for n, i in enumerate(answer):
            print('知识%d: %s' % (n + 1, i))

    print('\n----------------我是分割线----------------\n')
