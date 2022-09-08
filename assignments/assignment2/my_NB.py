import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha
        
        

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # Calculate P(yj) and P(xi|yj)        
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        # write your code below
        cols=X.columns.tolist()
        uniq_dict={}
        for i in cols:
            uniq_dict[i]=(X[i].unique().tolist())
            
        #dictionary for thr p_y values
        self.p_y = {}

        for clss in y.unique().tolist():
            self.p_y[clss] = (y[y==clss].count())
            
        self.dict_x_y={}
        
        for col in uniq_dict:
            self.dict_x_y[col]={}
            for col_val in uniq_dict[col]:
                self.dict_x_y[col][col_val]={}
                for clss in y.unique().tolist():
                    self.dict_x_y[col][col_val][clss]={}
                    p_x_y= X.loc[(y==clss)& (X[col]==col_val), col].count()
                    #p_y=X.loc[(y==clss), col].count()
                
                    #{independentvariable:{category in ind variable list: {dependent variable: prob value}}}
                    self.dict_x_y[col][col_val][clss]= (p_x_y+ self.alpha)/(self.p_y[clss]+ (len(uniq_dict[col])*self.alpha))

        #print (dict_x_y)
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code 
        self.predictions=[]
        
        dict_test={}

        for test_row in X.index:
    
            dict_test[test_row] = {}
    
            for clss in self.classes_:
        
                prob_y_x = self.p_y[clss]
        
                for col in X.columns:
            
                    value = X.loc[test_row,col]
            
                    if value in self.dict_x_y[col].keys():
                        prob_y_x *=  self.dict_x_y[col][value][clss]
                
                    else:
                        prob_y_x *= 1
                
                dict_test[test_row][clss] = prob_y_x
                self.probs_df = pd.DataFrame(dict_test)
                for col in  self.probs_df.columns:    
                    self.probs_df[col] = self.probs_df[col]/ self.probs_df[col].sum()
                self.probs_df = self.probs_df.T
                
                self.predictions= self.probs_df.idxmax(axis=1).to_list()
                
                
        return self.predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)                
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # write your code below
        self.probs=self.probs_df
        return self.probs



