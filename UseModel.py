import json
import random
import pickle
import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# -----NLTK downloads-----
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----Variables-----
model = load_model('./ModelResources/chatbot_model.h5')
model._make_predict_function()
intents = json.loads(open('./Data/BasicQueries.json').read())
words = pickle.load(open('./ModelResources/words.pkl','rb'))
classes = pickle.load(open('./ModelResources/classes.pkl','rb'))
lemmatizer = WordNetLemmatizer()

# -----Function for cleaning the sentences-----
def sentence_cleaning(sentence):
    words_of_sentence = nltk.word_tokenize(sentence)
    words_of_sentence = [lemmatizer.lemmatize(word.lower()) for word in words_of_sentence]

    return words_of_sentence

# -----Function for defining collection of words-----
def collection_of_words(sentence, words, show_details=True):
    words_of_sentence = sentence_cleaning(sentence)
    collection = [0]*len(words) 

    for s_word in words_of_sentence:
        for index,word in enumerate(words):
            if word == s_word: 
                collection[index] = 1
                if show_details:
                    print ("In the collection : %s" % w)

    return(np.array(collection))

# -----Function for predicting class-----
def class_prediction(sentence, model):
    predictions = collection_of_words(sentence, words, show_details=False)
    response = model.predict(np.array([predictions]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[index,result] for index,result in enumerate(response) if result>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    output_list = []

    for result in results:
        output_list.append({"intent": classes[result[0]], "probability": str(result[1])})
    return output_list

# -----Function for retrieving response-----
def response_retrieving(ints, intents_json):
    result = ""
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if(intent['tag'] == tag):
            result = random.choice(intent['dialog2'])
            break
    return result

def response_from_chatbot(string):
    ints = class_prediction(string, model)
    res = response_retrieving(ints, intents)
    return res

print(response_from_chatbot("Why are you laugh so much"))