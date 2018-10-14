import tweepy
import csv
import pandas as pd

token_dict={}

with open('data/twitter_api_keys.csv') as csv_file:
    csv_reader=csv.DictReader(csv_file, delimiter=',')
    for row in csv_reader:
    	for (k,v) in row.items():
            token_dict[k]=v

auth = tweepy.OAuthHandler(token_dict['consumer_key'], token_dict['consumer_secret'])
auth.set_access_token(token_dict['access_token'], token_dict['access_token_secret'])
api = tweepy.API(auth, wait_on_rate_limit=True)

# Open/Create a file to append data
csvFile = open('data/twitter_data_new.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#Google",count=100,
                           lang="en",
                           since="2018-04-03").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])