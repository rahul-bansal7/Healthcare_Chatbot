import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models,layers
from fastapi import FastAPI
import uvicorn
import nltk
import numpy as np
import re
import random
import time

# nltk.download('punkt') # downloading model to tokenize message
from nltk.tokenize import word_tokenize

# nltk.download('stopwords') # downloading stopwords
from nltk.corpus import stopwords

# nltk.download('wordnet') # downloading all lemmas of english language
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()

def clean_corpus(data):
  corpus=[]
  for i in data:
    review=re.sub('[^a-zA-Z]',' ',i)
    review=review.lower()
    review=review.split()
    review=[lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
  return corpus

app = FastAPI()

MODEL=tf.keras.models.load_model("C:\\Users\\Ratnesh\\Desktop\\Health_Care\\Intents\\model_creation\\model1.h5")

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

INTENT_NOT_FOUND_THRESHOLD = 0.40
def predict_intent_tag(message):
    message = clean_corpus([message])
    X_test = vectorizer.transform(message)
    y = model.predict(X_test.toarray())
    # if probability of all intent is low, classify it as noanswer
    if y.max() < INTENT_NOT_FOUND_THRESHOLD:
        return 'noanswer'
    prediction = np.zeros_like(y[0])
    prediction[y.argmax()] = 1
    tag = encoder.inverse_transform([prediction])[0][0]
    return tag


def get_intent(tag):
    # to return complete intent from intent tag
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return intent

@app.get("/chat")
async def chat():
  while True:
    # get message from user
    message = input('You: ')
    # predict intent tag using trained neural network
    tag = predict_intent_tag(message)
    # get complete intent from intent tag
    intent = get_intent(tag)
    # generate random response from intent
    response = random.choice(intent['responses'])
    print('Bot: ', response)
    # check if there's a need to perform some action
    if 'action' in intent.keys():
        action_code = intent['action']
        # perform action
        data = perform_action(action_code, intent)
        # get follow up intent after performing action
        followup_intent = get_intent(data['intent-tag'])
        # generate random response from follow up intent
        response = random.choice(followup_intent['responses'])
        # print randomly selected response
        if len(data.keys()) > 1:
            print('Bot: ', response.format(**data))
        else:
            print('Bot: ', response)
    # break loop if intent was goodbye
    if tag == 'goodbye':
        break

if __name__=="__main__":
  uvicorn.run(app,port=8000,host='localhost')