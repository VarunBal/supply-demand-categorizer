from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import to_database as db
import matplotlib.pyplot as plt
import re
import nltk
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

db_data = np.array(db.select_data())

stop_list = ['hai','hi','ji','thanks','thnxs','plz']
stop_list = ENGLISH_STOP_WORDS.union(stop_list)

org_documents = db_data[:,10]
documents = org_documents

vectorizer = TfidfVectorizer(stop_words=stop_list)

patterns = re.compile('available|avail|need|require|requirement',re.IGNORECASE)

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
print(final_documents)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])
X = pipeline.fit_transform(final_documents).todense()

##number of clusters
true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=100)
model.fit(X)

print(model.labels_)

###visualizer
pca = PCA(n_components=3).fit(X)
data2D = pca.transform(X)
plt.scatter(data2D[:,0], data2D[:,1],c=model.labels_)

centers2D = pca.transform(model.cluster_centers_)

plt.scatter(centers2D[:,0], centers2D[:,1], 
            marker='x', s=200, linewidths=3, c='r')
plt.show()

## 
##Y = vectorizer.transform(["need hotel for 3 people from in manali"])
##prediction = model.predict(Y)
##print(prediction)
## 
##Y = vectorizer.transform(["if anyone needs hotels in shimla contact me"])
##prediction = model.predict(Y)
##print(prediction)
