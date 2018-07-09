import pickle
import warnings

warnings.filterwarnings('ignore')

bye = '''
    　　 へ　　　　　／|
　　/＼7　　　 ∠＿/
　 /　│　　 ／　／
　│　Z ＿,＜　／　　 /`ヽ
　│　　　　　ヽ　　 /　　〉
　 Y　　　　　`　 /　　/
　ｲ●　､　●　　⊂⊃〈　　/
　()　 へ　　　　|　＼〈
　　>ｰ ､_　 ィ　 │ ／／
　 / へ　　 /　ﾉ＜| ＼＼
　 ヽ_ﾉ　　(_／　 │／／
　　7　　　　　　　|／
　　＞―r￣￣`ｰ―＿
。
    '''

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
                                        topn=5)

    print('\n----------------我是分割线----------------\n')
