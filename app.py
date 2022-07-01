# Importing Required Libraries
import json
from tkinter import *
import tensorflow as tf
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random

# Loading Data
data_file = open('intents.json').read()
intents  = json.loads(data_file)

# Loading Model
model = tf.keras.models.load_model('chatbot_model.h5')
# Unpickling Files
classes = pickle.load(open('classes.pkl', 'rb'))
words = pickle.load(open('words.pkl', 'rb'))
# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

root = Tk()
root.title('Health Care Chatbot')
root.geometry("400x500")
root.resizable(height=False, width=False)

chat_history = Text(root, bd=0, bg='white', font='arial')
chat_history.config(state=DISABLED)


def clean_up_msg(message):
    # ignore punctuation from words
    ignore_words = ['?',',','!','.']
    tokenized_words = nltk.word_tokenize(message)
    clean_words = [lemmatizer.lemmatize(w.lower()) for w in tokenized_words if w not in ignore_words]
    return clean_words

def bow(message, words):
    clean_words = clean_up_msg(message)
    bag = [0] * len(words)
    for clean_word in clean_words:
        for idx, word in enumerate(words):
            if word == clean_word:
                bag[idx] = 1
    return np.array(bag)

def predict_class(message):
    bag_of_words = bow(message, words)
    res = model.predict(np.array([bag_of_words]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[idx, result] for idx, result in enumerate(res) if result > ERROR_THRESHOLD]
    # Sort by Probability
    results.sort(key= lambda x: x[1], reverse= True)
    intent_with_probability = [{'intent': classes[res[0]], 'probabilty' : str(res[1])} for res in results]
    return intent_with_probability

def getResponse(ints_with_probability):
    if ints_with_probability and ints_with_probability[0]['probabilty']:
        tag = ints_with_probability[0]['intent']
        list_of_intents = intents['intents']

        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        resp = ["Sorry, can't understand you","Sorry,i don't get it" ,"Please give me more info", "Not sure I understand","I missed what you said"]
        result = random.choice(resp)
    return result

def chatbot_response(msg):
    ints_with_probability = predict_class(msg)
    res = getResponse(ints_with_probability)
    return res


def send():
    msg = TextEntryBox.get('1.0', 'end-1c').strip()
    TextEntryBox.delete('1.0', END)
    if msg != '':
        chat_history.config(state= NORMAL)
        chat_history.insert('end', 'You: ' + msg + '\n\n')
        res = chatbot_response(msg)
        chat_history.insert('end', 'Bot: ' + res + '\n\n')
        chat_history.config(state= DISABLED)
        chat_history.yview('end')


SendButton = Button(root,
                    font=('Arial', 14, 'bold'),
                    text='Send',
                    bg='#044709',
                    activebackground="#2a612e",
                    fg="#ffffff",
                    command=send)

#Define a function to clear the content of the text widget
def click(event):
   TextEntryBox.configure(state=NORMAL)
   TextEntryBox.delete('1.0', END)
   TextEntryBox.unbind('<Button-1>', clicked)         

# Create Text widget
TextEntryBox = Text(root, bd=0, bg='#ede1e1', font='arial')
TextEntryBox.insert(END, "Please Type Here...")

chat_history.place(x=6, y=6, height=386, width = 386)
TextEntryBox.place(x=6, y=400, height=80, width=265)
SendButton.place(x=275, y=400, height=80, width=125)

#Bind the Entry widget with Mouse Button to clear the content
clicked = TextEntryBox.bind('<Button-1>', click)

root.mainloop()
