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
import pandas as pd
pd.set_option('display.max_columns', 999)


df = pd.read_csv('dftokenscleaned.csv')
df = df.drop(columns=['Unnamed: 0'])

df


from wordcloud import WordCloud
from os import path, getcwd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator



tokenlist = df.tokens.tolist()
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
