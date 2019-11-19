from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import re
import nltk
import numpy as np

org_documents = ["avail avail available free",
             "available avail",
             "need require",
             "available available free",
             "requirement need",
             "available avail require free",
             "need requirement require",
             "require"]

stop_list = ['hai','hi','ji','thanks','thnxs','plz']
stop_list = ENGLISH_STOP_WORDS.union(stop_list)

documents = org_documents

vectorizer = TfidfVectorizer(stop_words=stop_list)

patterns = re.compile('available|avail|need|require|requirement|free',re.IGNORECASE)

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if patterns.search(token):
            filtered_tokens.append(token)
    return filtered_tokens

final_documents = []
for i in documents:
    allwords_tokenized = tokenize_only(i)
    if ' '.join(allwords_tokenized) != '':
        final_documents.append(' '.join(allwords_tokenized))
##print(final_documents)

X = vectorizer.fit_transform(final_documents)


##number of clusters
true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=100)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
print("\n")
print("Prediction")

print(model.labels_)
try:
    joblib.dump(vectorizer,'vectorizer.vec')
    print('vectorizer saved')
    joblib.dump(model,'saved_model.mdl')
    print('model saved')
except Exception as e:
    print(e)
