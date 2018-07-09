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

with open('./model_chat.pkl', mode='rb') as f:
    chatbot_try = pickle.load(f)

print('我是山寨版小黄鸡，很高兴见到你！\n\n')
state = True
while state:
    # print('请输入您的问题（范例：谁负责全国医疗器械监督管理工作？）\n离开请输入\"quit\"：')
    print('你说：')
    ask = input()
    if ask == 'quit':
        print(bye)
        break
    else:
        print('\n小黄鸡说')
        answer = chatbot_try.get_answer(ask=ask,
                                        sample=80000,
                                        similarity='cos',
                                        modify=False,
                                        threshold=-1,
                                        topn=3)
        print('\n')

    # print('\n----------------我是分割线----------------\n')



