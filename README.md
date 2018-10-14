# Twitter_Sentiment_LSTM
This project uses an LSTM network to predict tweet sentiment to a 72% accuracy.

## Dependencies:
- Keras
- Pandas
- Numpy
- Sklearn
- Wordcloud
- BeautifulSoup
- NLTK
- Tweepy

The project uses an LSTM network to predict sentiment of tweets. The dataset used is include in the repo and consists of 1.6 million tweets. The original source for this data was found on kaggle --->  https://www.kaggle.com/kazanova/sentiment140

Download the dataset and rename the csv file to tweets.csv, put this file inside a new folder named data and run the preprocessing python file to preprocess the tweets for training.

So far I have managed to obtain a 72% accuracy with my model. I believe this is due in part to the vast amount of neutral words found in the data, some of which I have removed during preprocessing. During preprocessing I removed unusual characters that were not UTF-8 compliant. 
