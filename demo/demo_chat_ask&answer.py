import pickle
import warnings
from random import sample
warnings.filterwarnings('ignore')

if __name__ == '__main__':
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
    ask_old = None
    while state:
        # print('请输入您的问题（范例：谁负责全国医疗器械监督管理工作？）\n离开请输入\"quit\"：')
        print('你说：')
        ask = input()
        if ask == ask_old:
            print('\n小黄鸡说：')
            print(sample(['干嘛说一样的话？',
                          '干嘛又说一遍...',
                          '为什么要重复啊...',
                          '我不喜欢重复哎...'], 1)[0] + '\n')
        else:
            if ask == 'quit':
                print(bye)
                break
            else:

                answer = chatbot_try.get_answer(ask=ask,
                                                sample=500000,
                                                similarity='cos',
                                                modify=False,
                                                threshold=0,
                                                topn=5,
                                                process_num=2)
                print('\n小黄鸡说：\n' + answer + '\n')

                # print('\n----------------我是分割线----------------\n')
        ask_old = ask

