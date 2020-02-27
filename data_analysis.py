import pandas as pd
pd.set_option('display.max_columns', 999)


df = pd.read_csv('unique_listings.csv')
df = df.drop(columns='Unnamed: 0')

df


df[df['category_id'] == 69150367.0]['category_path'].head(1)
df.category_path.value_counts().head(50)
df.category_id.value_counts().head(10)
df.taxonomy_id.value_counts().head(10)
df.taxonomy_path.value_counts().head(50)


df[df['category_path'] == "['Supplies']"]['taxonomy_path'].unique()

df.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,7))
plt.xticks(rotation=90)
ax = sns.countplot(x='taxonomy_path', data=df[df.groupby('taxonomy_path')['taxonomy_path'].transform('size') >= 50])



plt.figure(figsize=(15,7))
plt.xticks(rotation=90)
ax = sns.countplot(x='category_path', data=df[df.groupby('category_path')['category_path'].transform('size') >= 100])


df[df['views']==df['views'].max()]


df.dtypes

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from surprise.prediction_algorithms import knns
from surprise.similarities import cosine, msd, pearson
from surprise import accuracy
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise.model_selection import GridSearchCV
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string, re
from nltk.tokenize import RegexpTokenizer
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('unique_listings.csv')
df = df.drop(columns=['Unnamed: 0', 'recipient', 'occasion', 'currency_code', 'quantity', 'language'])

df

dftokens = df.copy()

stringcol = list(df.select_dtypes(['object']).columns)
stringcol.remove('url')
stringcol


for col in stringcol:
    dftokens[col] = dftokens[col].astype(str)
    dftokens[col] = dftokens[col].str.lower()


tokenizer = RegexpTokenizer(r'(?u)(?<![@])#?\b\w\w+\b')
sw_list = stopwords.words('english')
sw_list += ["''", '""', '...', '``', '’', '“', '’', '”', '‘', '‘', '©', 'com', '#39', 'etsy', 'www', 'https']
sw_set = set(sw_list)

def process_text(col):
    tokens = tokenizer.tokenize(col)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in sw_set]
    return stopwords_removed


for col in stringcol:
    dftokens[col] = list(map(process_text, dftokens[col]))


dftokens['tokens'] = (dftokens.title + dftokens.tags + dftokens.category_path + dftokens.taxonomy_path).map(set).map(list)
dftokens['tokens'] = dftokens.tokens.apply(', '.join)
dftokens

dftokens.to_csv('dftokens.csv')

dftokens

dftokens = dftokens.drop(columns=['user_id', 'title', 'description'])

dftokens.tokens = dftokens.tokens.str.replace(r',', '')

dftokens.to_csv('dftokenscleaned.csv')




from wordcloud import WordCloud
from os import path, getcwd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

dftokens

tokenlist = dftokens.tokens.tolist()
tokenlist



tokenlist2 = []
for i in tokenlist:
    tokenlist2.extend(i.split())


len(tokenlist2)

tokenset = set(tokenlist2)
tokenset


from nltk.probability import FreqDist
tokenfreq = FreqDist(tokenlist2)

tokenfreq

real_bar_words = [x[0] for x in tokenfreq.most_common(25)]
real_bar_words
real_bar_counts = [x[1] for x in tokenfreq.most_common(25)]
real_bar_counts



real_dictionary = dict(zip(real_bar_words, real_bar_counts))
wordcloud = WordCloud(colormap='Spectral').generate_from_frequencies(real_dictionary)

plt.figure(figsize=(7, 7), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
# plt.savefig('realtop25.png')

plt.show()











count = CountVectorizer()
count_matrix = count.fit_transform(dftokens['tokens'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)



indices = pd.Series(dftokens.index)


def recommend(token, cosine_sim = cosine_sim):
    recommend_listing = []
    idx = indices[indices == results[row['tokens']]].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top10 = list(score_series.iloc[1:11].index)
    for i in top10:
        recommend_listing.append(list(dftokens.index)[i])
    return recommend_listing


score_series
recommend('baby')




results = {}
for idx, row in dftokens.iterrows():
   similar_indices = cosine_sim[idx].argsort()[:-100:-1]
   similar_items = [(cosine_sim[idx][i], dftokens['url'][i]) for i in similar_indices]
   results[row['tokens']] = similar_items[1:]

list(results.items())[0]

def item(token):
  return dftokens[dftokens['tokens'].str.contains(token)].sort_values(by='num_favorers', ascending=False)['url'].tolist()[0]

def time(token):
    return dftokens.loc[dftokens[]]



print(item('straw'))


# Just reads the results out of the dictionary.
def recommend(token, num):
    print("Recommending " + str(num) + " products similar to " + item(token) + "...")
    print("-------")
    recs = results[item(token)][:num]
    for rec in recs:
       print("Recommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")")




recommend('baby', 10)




recommend_df = dftokens[dftokens['tokens'].str.contains('baby')].sort_values(by='num_favorers', ascending=False)['url'].tolist()[0]
recommend_df
