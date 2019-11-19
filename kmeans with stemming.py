from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import to_database as db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
 
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

db_data = np.array(db.select_data())

stop_list = ['hai','hi','ji','thanks','thnxs','plz']
stop_list = ENGLISH_STOP_WORDS.union(stop_list)

documents = db_data[:,10]

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in documents:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame)

##vectorizer = TfidfVectorizer(stop_words=stop_list)
##X = vectorizer.fit_transform(documents)
##
####print(X)
####print(vectorizer.idf_)
####print(vectorizer.stop_words_)
##
####number of clusters
##true_k = 2
##
##model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=100)
##model.fit(X)
##
##print("Top terms per cluster:")
##order_centroids = model.cluster_centers_.argsort()[:, ::-1]
##terms = vectorizer.get_feature_names()
##for i in range(true_k):
##    print("Cluster %d:" % i),
##    for ind in order_centroids[i, :10]:
##        print(' %s' % terms[ind]),
##    print
##print("\n")
##print("Prediction")
##
##print(model.labels_)
## 
##Y = vectorizer.transform(["need hotel for 3 people from in manali"])
##prediction = model.predict(Y)
##print(prediction)
## 
##Y = vectorizer.transform(["if anyone needs hotels in shimla contact me"])
##prediction = model.predict(Y)
##print(prediction)
