import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

class my_model():

    def fit(self, X, y):
        # do not exceed 29 mins
        X.replace("#NAME?", " ")
        X.fillna(" ",inplace=True)
        
        X['text_col'] = X['title'] + " " + X['location'] + " " + X['description'] + " "+ X['requirements']
        X['text_col'] = X['text_col'].str.lower()
        
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2')
        XX = self.preprocessor.fit_transform(X["text_col"])
        
        base = SGDClassifier(class_weight='balanced', random_state = 0)

        param_grid = {"loss":["hinge", "log", "perceptron"], 'penalty': ["l2", "elasticnet"], 'alpha': [0.0001,0.001,0.01]}
        self.clf = GridSearchCV(estimator=base, param_grid=param_grid, scoring='f1', cv=5)
        
        self.clf.fit(XX,y)

        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X.replace("#NAME?", " ")
        X.fillna(" ",inplace=True)
        
        X['text_col'] = X['title'] + " " + X['location'] + " " + X['description'] + " "+ X['requirements']
        X['text_col'] = X['text_col'].str.lower()
        
        XX = self.preprocessor.transform(X["text_col"])
        
        predictions = self.clf.predict(XX)
        return predictions

