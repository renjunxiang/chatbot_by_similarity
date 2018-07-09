from chatbot import chatbot
import pickle

texts = load_data(type='knowledge')

chatbot_try = chatbot()
chatbot_try.train(texts=texts, mode='knowledge')
with open('./model_word_document.pkl', mode='wb') as f:
    pickle.dump(chatbot_try, f)
