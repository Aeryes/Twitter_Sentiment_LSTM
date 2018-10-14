import pandas as pd  
import numpy as np
import collections

import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

# Function used to clean unprocessed tweets.
def tweet_cleaner(text):
    soup = BeautifulSoup(text, features='lxml')
    souped = soup.get_text()

    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped

    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    
    # Remove words that have little meaning in the sentiment to improve accuracy, words found using wordcloud.
    word_list = ['haha', 'tell', 'made', 'look', 'one', 'now', 'tonight', 'wish', 'last', 'night', 'morning', 'know', 'though', 'today', 'still', 'way', 'week', 'al', 'got', 'ye', 'time', 'thing',
     'cl', 'even', 'summer', 'day','make']
    for word in word_list:
        stripped = re.sub(word, '',stripped)

    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]

    return (" ".join(words)).strip()

if __name__ == '__main__':
    # Open/Create a file to append data and preprocess tweets.
    cols = ['sentiment', 'id','date','query_string','user','text']

    tweet_file = pd.read_csv('data/tweets.csv', encoding='ISO-8859-1', names=cols)
    tweet_file.drop(['id','date','query_string','user'],axis=1,inplace=True)
    data = []

    for text in tweet_file['text']:
        feature_list = tweet_cleaner(text)
        data.append(feature_list)
    
    print('Creating Tweet Text CSV...')
    clean_df = pd.DataFrame(data, columns=['text'])
    clean_df.to_csv('data/clean_tweet.csv',index=False, encoding='utf-8')
    print('Tweet Text CSV Created!')
    

    # Add Sentiment column to the file.
    new_column = tweet_file['sentiment']
    data_sentiment = []

    for line in new_column:
        line = re.sub('[4]', '1', str(line))
        data_sentiment.append(line)
    
    print('Creating Sentiment CSV...')
    clean_sentiment = pd.DataFrame(data_sentiment, columns=['sentiment'])
    clean_sentiment.to_csv('data/tweet_sentiment.csv', index=False)
    print('Sentiment CSV Created!')
