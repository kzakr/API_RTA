
import pickle
from math import log10
import flask
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import sklearn

import random

class Perceptron():
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        # self.w_ = np.zeros(1+X.shape[1])
        
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] 
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #print(xi, target)
                update = self.eta*(target-self.predict(xi))
                #print(update)
                self.w_[1:] += update*xi
                self.w_[0] += update
                #print(self.w_)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0,1,-1)

# Create a flask
app = Flask(__name__)

# Create an API end point
@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():

    # sepal length
    sepal_length = float(request.args.get('sl'))
    # sepal width
    #sepal_width = float(request.args.get('sw'))
    # petal length
    petal_length = float(request.args.get('pl'))
    # petal width
    #petal_width = float(request.args.get('pw'))

    # The features of the observation to predict
    #features = [sepal_length,
    #            sepal_width,
    #            petal_length,
    #           petal_width]
    
    features1 = [sepal_length,
                petal_length]
    features = np.array([[sepal_length],
                [petal_length]]).reshape(-1, 1)
    
    
    #print(features)
    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
    
    # Predict the class using the model
    #predicted_class = int(model.predict(features))
    predicted_class = [int(x) for x in (model.predict(features))]
    
    #return f"costam {predicted_class}"
    # Return a json object containing the features and prediction
    return jsonify(features=features1, predicted_class=predicted_class)

if __name__ == '__main__':
    #docker build -t modelml
    
#docker run -p 5000:5000 test_hello
    app.run(host="0.0.0.0", debug=True)
