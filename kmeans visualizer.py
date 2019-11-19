from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import to_database as db

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

db_data = db.select_data()

stop_list = ['hai','hi','ji','thanks','thnxs']
stop_list = ENGLISH_STOP_WORDS.union(stop_list)

documents = []
for i in range(len(db_data)):
    documents.append(db_data[i][3])

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])
X = pipeline.fit_transform(documents).todense()

#number of clusters
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


