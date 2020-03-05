from surprise.similarities import cosine, msd, pearson
from surprise import accuracy
from sklearn.model_selection import train_test_split
import numpy as np
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
import sys
from tqdm import tqdm
import streamlit as st

nlp = spacy.load("en_core_web_md")
df = pd.read_csv('dfclusterid.csv')
df = df.drop(columns=['Unnamed: 0'])
tqdm.pandas()
df['spacy'] = df.tokens.progress_apply(lambda x: nlp(x))

df


df.loc[df['cluster'] == 18]


termdf = pd.read_csv('clustertermdf.csv')
termdf = termdf.drop(columns=['Unnamed: 0'])
termdf['clusterterms'] = termdf.clusterterms.progress_apply(lambda x: nlp(x))


termdf


def find_arg_cluster(*argv):
    cluster_sim = []
    for i in range(25):
        cluster_sim.append((nlp(str(argv)).similarity(termdf.clusterterms[i])))
    return (pd.DataFrame(cluster_sim)).idxmax().item()

find_arg_cluster('bridal shower')

def get_index(*argv):
    irange = []
    irange.extend(df.loc[df['cluster'] == (find_arg_cluster(*argv))].index)
    return irange

get_index('bridal shower')

def sim_score(*argv):
    index_scores = {}
    for i in get_index(*argv):
        index_scores.update({i: (nlp(str(argv))).similarity(df.spacy[i])})
    return index_scores


sim_score('bridal shower')

def top10(*argv):
    index_scores = sim_score(*argv)
    N = 10
    top = nlargest(N, index_scores, key = index_scores.get)
    return df.iloc[top]['url'].tolist()


top10('bridal shower')
