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
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import spacy
import sys
from tqdm import tqdm
from collections import Counter
from heapq import nlargest
import pickle
pd.set_option('display.max_columns', 999)


tqdm.pandas()
nlp = spacy.load("en_core_web_md")
df = pd.read_pickle('dfspacy.pkl')
df = df.drop(columns=['tags', 'category_path', 'taxonomy_path'])


doctry = df.spacy[10464]
doctry2 = nlp('hair scrunchie')
doctry.similarity(doctry2)



# count = CountVectorizer()
# count_matrix = count.fit_transform(df['tokens'])
# cosine_sim = cosine_similarity(count_matrix, count_matrix)
#
# count_matrix
#
#
#
#
#
#
# model = KMeans(n_clusters=25, init='k-means++', max_iter=500, n_init=15)
# model.fit(count_matrix)
#
# np.set_printoptions(threshold=sys.maxsize)
# model.labels_


# pickle.dump(model, open('kmeansclustermodel.sav', 'wb'))






dfclusters = pd.DataFrame(model.labels_, columns=['cluster'], index=df.spacy)
dfclusters.to_csv('dfclusters.csv')

df2 = df.copy()
df2 = df2.merge(dfclusters, on=df2.spacy)
df2 = df2.drop(columns=['key_0', 'category_id', 'price', 'category_path_ids', 'shop_section_id', 'views', 'num_favorers', 'taxonomy_id'])
dfclusterid = df2

dfclusterid.to_csv('dfclusterid.csv')

dfclusterid.head(41)


# print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = count.get_feature_names()
clusterdict = {}
for i in range(25):
    for ind in order_centroids[i, :15]:
        clusterdict.update({('%s' % terms[ind]): (i)})



clusterdict

pickle.dump(clusterdict, open('clusterdict', 'wb'))

clustertermlist = []
for i in range(25):
    clustertermlist.append(' '.join([k for k,v in clusterdict.items() if v == i]))

clustertermlist

clustertermdf = pd.DataFrame(clustertermlist)

clustertermdf.columns = ['clusterterms']
clustertermdf['clusterterms'] = clustertermdf.clusterterms.progress_apply(lambda x: nlp(x))

clustertermdf.to_csv('clustertermdf.csv')


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = count.get_feature_names()
for i in range(25):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print





dfclusterid.loc[dfclusterid['cluster'] == 8].index






def find_arg_cluster(*argv):
    cluster_sim = []
    for i in range(25):
        cluster_sim.append((nlp(str(argv)).similarity(clustertermdf.clusterterms[i])))
    return (pd.DataFrame(cluster_sim)).idxmax().item()

find_arg_cluster('baby')

def get_index(*argv):
    irange = []
    irange.extend(dfclusterid.loc[dfclusterid['cluster'] == (find_arg_cluster(*argv))].index)
    return irange

get_index('baby')

def sim_score(*argv):
    index_scores = {}
    for i in get_index(*argv):
        index_scores.update({i: (nlp(str(argv))).similarity(dfclusterid.spacy[i])})
    return index_scores


sim_score('baby')


def top10(*argv):
    index_scores = sim_score(*argv)
    N = 10
    top = nlargest(N, index_scores, key = index_scores.get)
    return dfclusterid.iloc[top]['url'].tolist()



top10('baby')
