from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
import numpy as np
import pandas as pd

# Import data and run it through the pre-trained model, model acc is 70%.
#csv_file = pd.read_csv()
#csv_file.dropna(inplace=True)
#csv_file.reset_index(drop=True, inplace=True)
tweet_data = 'bad people do bad things i hate this shit bummer'

# Integer encode the sentiments.
vocab_size = 200
encoded_docs = [one_hot(d, vocab_size) for d in tweet_data] 
max_length = 120
X_new = sequence.pad_sequences(encoded_docs, maxlen=max_length) 

def run_JSON_model(filename_model, filename_weights, data):
    # Load json and create model
    json_file = open(filename_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Load weights into new model
    loaded_model.load_weights(filename_weights)
    print("Loaded model from disk...")

    # Predict the new data.
    prediction = loaded_model.predict(data)[0]

    if prediction > 0.5:
        print(f'{tweet_data}\nShows a Positive Sentiment of {prediction}')
    else:
        print(f'{tweet_data}\nShows a Negative Sentiment of {prediction}')

run_JSON_model('model/model.json', 'model/model.h5', X_new)