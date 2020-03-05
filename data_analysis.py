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
