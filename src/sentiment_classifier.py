from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

twitter_data = pd.read_csv('data/clean_tweet.csv')
twitter_sentiment = pd.read_csv('data/tweet_sentiment.csv')
twitter_data.dropna(axis=1)
twitter_data.reset_index(drop=True)
tweet_text = twitter_data['text']
tweet_labels = twitter_sentiment['sentiment']

X_train, X_test, y_train, y_test = train_test_split(tweet_text, tweet_labels, test_size = 0.2, random_state = 0)

# Integer encode the sentiments.
vocab_size = 200
X_train = [one_hot(d, vocab_size) for d in X_train]
X_test = [one_hot(d, vocab_size) for d in X_test]

# Setting up the training data. 
max_length = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_length) 
X_test = sequence.pad_sequences(X_test, maxlen=max_length)  

# Build the model  
input_dim = 501
output_dim = 32 
model = Sequential() 
model.add(Embedding(input_dim, output_dim, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=1) 

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights("model/model.h5")
print("Saved model to disk")