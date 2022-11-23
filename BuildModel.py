import nltk
import random
import json
import pickle
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

# -----Variables-----
words=[]
classes = []
documents = []
words_to_ignore = ["?", "!", "^", "$", "@"]
data = open('./Data/BasicQueries.json').read()
intents = json.loads(data)
lemmatizer = WordNetLemmatizer()

# -----NLTK downloads-----
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----Data preprocessing-----
for intent in intents['intents']:
    for dialog1 in intent['dialog1']:

        word = nltk.word_tokenize(dialog1)
        words.extend(word)
        documents.append((word, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# -----lemmatize-----
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in words_to_ignore]
words = sorted(list(set(words)))

# -----Class sorting-----
classes = sorted(list(set(classes)))
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "Lemmatized words (Unique)", words)

# -----Create words and classes files-----
pickle.dump(words,open('./ModelResources/words.pkl','wb'))
pickle.dump(classes,open('./ModelResources/classes.pkl','wb'))

# -----Create training set-----
training_set = []
empty_output = [0] * len(classes)
for doc in documents:
    collection = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for word in words:
        collection.append(1) if word in pattern_words else collection.append(0)
    
    row_output = list(empty_output)
    row_output[classes.index(doc[1])] = 1
    
    training_set.append([collection, row_output])

# Create np array by shuffling features
random.shuffle(training_set)
training_set = np.array(training_set)

# Create training and testing data sets
trainX = list(training_set[:,0])
trainY = list(training_set[:,1])

# Create neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation='softmax'))

# Compile the created model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Fit and save the created model
model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('./ModelResources/chatbot_model.h5')