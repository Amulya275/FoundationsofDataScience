import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p
        self.predictions=[]
        

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X_train=X
        self.y_train=y
        
        # write your code below
        return
    
    def minkowski(self,train,test,p):
        sum=0
        list_diff = list(map(lambda x, y: abs(x - y), train,test))
        for i in list_diff:
            sum += i**p
        dist = sum**(1/p)
        return dist

    def norm_data(self,data_list):
        s=0
        for i in data_list:
            s+=i**2
        return (s**0.5)

    def cosine(self,train,test):
        norm_x=self.norm_data(train)
        norm_y=self.norm_data(test)
        cos= np.dot(train,test)/(norm_x*norm_y)
        return (1-cos)

    def predict(self, X):
        
        
      
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        
        probs = self.predict_proba(X)
        predictions = probs.idxmax(1).tolist()
        
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        self.X_test=X
        self.dict_test={}
        distance=0
        
        for test_row in self.X_test.index:
            
            self.dict_test[test_row]=[]
            
            for train_row in self.X_train.index:
            
                if self.metric=="minkowski":
                    distance = self.minkowski(self.X_train.iloc[train_row,:].tolist(),self.X_test.iloc[test_row,:].tolist(),self.p)
                
                elif self.metric=="euclidean":
                    distance = self.minkowski(self.X_train.iloc[train_row,:].tolist(),self.X_test.iloc[test_row,:].tolist(),2)
                
                elif self.metric=="manhattan":
                    distance = self.minkowski(self.X_train.iloc[train_row,:].tolist(),self.X_test.iloc[test_row,:].tolist(),1)
                
                elif self.metric=="cosine":
                    distance = self.cosine(self.X_train.iloc[train_row,:].tolist(),self.X_test.iloc[test_row,:].tolist())
                
                self.dict_test[test_row].append(distance)   
        
        
        self.counter_dict={}
        
        for i in self.X_test.index:
            
            dict_v=[self.y_train[j] for j in np.argsort(self.dict_test[i])[:self.n_neighbors]]
            c=Counter(dict_v)
            self.counter_dict[i] = c
    
        dict_pred = {}
        
        for i in self.X_test.index:
            
            dict_pred[i] = {}
            
            for j in self.classes_:
            
                dict_pred[i][j]=self.counter_dict[i][j]/self.n_neighbors
                
        probs = pd.DataFrame(dict_pred).T
        return probs



