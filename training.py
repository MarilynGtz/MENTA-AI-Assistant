import random
import json
import pickle
import numpy as np
import tensorflow as tf



import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

#intents = json.load(open('intents.json').read())
# Load intents from JSON file
with open('intents.json', 'r', encoding = 'utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Process intents and tokenize patterns (Tokenize separate words ex: i like pizza > 'i', 'like', 'pizza')
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and sort them, ex: dog to dogs, walk to walked etc
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes to pickle files (converts objects to bytes)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# Create training data
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert training data to numpy array
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

max_length = max(len(sentence) for sentence in trainX)


# Load words and classes from pickle files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Define the model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(max_length,), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Configure the optimizer and compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(trainX, trainY, epochs=600, batch_size=8, verbose=1)
model.save('chatbot_model.h5') # Save the trained model
print('Done')