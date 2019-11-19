from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import to_database as db
import matplotlib.pyplot as plt
import re
import numpy as np
 
##documents = ["Need ertiga Delhi to vrindaban Today 9 PM",
##             "I Required 5 Night 6 days Bangkok packeg no of pax 2 Date of journey Jan 1st week 2018",
##             "I need Tempo Travels tomorrow 7:00 am Manali to Delhi Drop call me   8894042702 Guru Dutt",
##             "Goanparadisetravels@gmail.com",
##             "Prices from 3500-5000",
##             "Need your support to increase my sales business. Have Great day.",
##             "Hello friend if any one has good delas in Five star hotel in delhi please call me need 2 rooms for 6th october area CP. Call Vivesh katyani. Holidays Bookers. +91 9910560003",
##             "Special Airfare AMD-DEL-AMD @ 12000/seat 21-27 OCT (09 seats) 22-28 OCT (28 seats) AMD-BLR-AMD @ 12000/seat 20-26 OCT (02 seats) BOM-AMD @ 3500/seat 25 OCT (12 seats) 26 OCT (12 seats) 27 OCT (12 seats) For bookings, contact: TICKET N TRIP Call: 9227555566 / 079-22863540 E-mail: info@ticketntrip.com Web: www.ticketntrip.com "]


db_data = np.array(db.select_data())

stop_list = ['hai','hi','ji','thanks','thnxs','plz']
stop_list = ENGLISH_STOP_WORDS.union(stop_list)

documents = db_data[:,10]

vectorizer = TfidfVectorizer(stop_words=stop_list)
X = vectorizer.fit_transform(documents)

##print(X)
##print(vectorizer.idf_)
##print(vectorizer.stop_words_)

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
 
Y = vectorizer.transform(["need hotel for 3 people from in manali"])
prediction = model.predict(Y)
print(prediction)
 
Y = vectorizer.transform(["if anyone needs hotels in shimla contact me"])
prediction = model.predict(Y)
print(prediction)
