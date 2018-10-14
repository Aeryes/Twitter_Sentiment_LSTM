import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

tweets = pd.read_csv('data/clean_tweet.csv')
tweets.dropna(axis=1)

#all_words = ' '.join([text for text in tweets['text']])
all_values = ','.join(str(v) for v in tweets['text'])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_values)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

for item in tweets['text']:
	if isinstance(item, float):
		print(item)
	else:
		continue
