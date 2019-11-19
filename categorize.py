from sklearn.externals import joblib
##import to_database as db
import numpy as np

##Kmeans model and vectorizer name to load
model_name = 'saved_model.mdl'
vectorizer_name = 'vectorizer.vec'

try:
    model = joblib.load(model_name)
    vectorizer = joblib.load(vectorizer_name)
except Exception as e:
    print(e)

def check_query(query):
    try:
##        ##fetch data from db
##        db_data = np.array(db.select_data())
##        query = db_data[:,3]
        
        Y = vectorizer.transform([query])
        
        if (Y.nnz==0):
            return 'unassigned'
        
        predictions = model.predict(Y)
        
        for prediction in predictions:
            if prediction == 0:
                return 'client'
            elif prediction == 1:
                return 'vendor'
        
    except Exception as e:
        print(e)

if __name__ == '__main__':
    result = check_query('i need 2 rooms')
    print(result)
