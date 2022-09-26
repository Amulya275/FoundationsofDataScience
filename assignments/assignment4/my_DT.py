import pandas as pd
import numpy as np
from collections import Counter


class my_DT:

    def __init__(self, criterion="gini", max_depth=8, min_impurity_decrease=0, min_samples_split=2):
        # criterion = {"gini", "entropy"},
        # Stop training if depth = max_depth
        # Only split node if impurity decrease >= min_impurity_decrease after the split
        # Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        # Only split node with >= min_samples_split samples
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = int(min_samples_split)


    def impurity(self, labels):
        # Calculate impurity (unweighted)
        # Input is a list (or np.array) of labels
        # Output impurity score
        stats = Counter(labels)
        N = float(len(labels))
        total = sum(stats.values(), 0)
        sum_values=0
        if self.criterion == "gini":
            # Implement gini impurity
            for key in stats:
                stats[key] /= total
                sum_values+=stats[key]**2
            gini = 1-sum_values
            #print("gini:" ,gini )
            impure=gini
        

        elif self.criterion == "entropy":
            # Implement entropy impurity
            for key in stats:
                stats[key] /= total
                sum_values+=(-stats[key]*np.log2(stats[key]))
            entropy = sum_values
            print("entropy:",entropy)
            impure=entropy
            
        else:
            raise Exception("Unknown criterion.")
        return impure;

    def find_best_split(self, pop, X, labels):
        # Find the best split
        # Inputs:
        #   pop:    indices of data in the node
        #   X:      independent variables of training data
        #   labels: dependent variables of training data
        # Output: tuple(best feature to split, weighted impurity score of best split, splitting point of the feature, [indices of data in left node, indices of data in right node], [weighted impurity score of left node, weighted impurity score of right node])
        ######################
        print(X.keys())
        max_gain = 0
        best_feature = None
        best_value = None
        
        def test_split(value, dataset):
            left=[]
            right = []
            for i in dataset:
                if i<value:
                    left.append(i)
                else:
                    right.append(i)
            return left,right
        
        def gain(groups, classes):
            n = float(sum([len(group) for group in groups]))
            gain_index= 0.0
            for  group in groups:
                size = float(len(group))
                if size==0:
                    continue
                gain_index+= (1.0 - (self.impurity(labels))) * (size / n)
                gain=self.impurity(self.classes_)-gain_index
            return gain
 
    
        for feature in X.keys():
            cans = np.array(X[feature][pop])
            for i in cans:
                groups =test_split(i,cans)
                gain_val=gain(groups,labels)
                print(feature,i,gain_val)
                
        return best_feature

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        labels = np.array(y)
        N = len(y)
        ##### A Binary Tree structure implemented in the form of dictionary #####
        # 0 is the root node
        # node i have two childen: left = i*2+1, right = i*2+2
        # self.tree[i] = tuple(feature to split on, value of the splitting point) if it is not a leaf
        #              = Counter(labels of the training data in this leaf) if it is a leaf node
        self.tree = {}
        # population keeps the indices of data points in each node
        population = {0: np.array(range(N))}
        # impurity stores the weighted impurity scores for each node (# data in node * unweighted impurity). 
        # NOTE: for simplicity reason we do not divide weighted impurity score by N here.
        impurity = {0: self.impurity(labels[population[0]]) * N}
        #########################################################################
        level = 0
        nodes = [0]
        while level < self.max_depth and nodes:
            # Breadth-first search to split nodes
            next_nodes = []
            for node in nodes:
                current_pop = population[node]
                current_impure = impurity[node]
                if len(current_pop) < self.min_samples_split or current_impure == 0:
                    # The node is a leaf node
                    self.tree[node] = Counter(labels[current_pop])
                else:
                    # Find the best split using find_best_split function
                    best_feature = self.find_best_split(current_pop, X, labels)
                    if best_feature and (current_impure - best_feature[1]) > self.min_impurity_decrease * N:
                        # Split the node
                        self.tree[node] = (best_feature[0], best_feature[2])
                        next_nodes.extend([node * 2 + 1, node * 2 + 2])
                        population[node * 2 + 1] = best_feature[3][0]
                        population[node * 2 + 2] = best_feature[3][1]
                        impurity[node * 2 + 1] = best_feature[4][0]
                        impurity[node * 2 + 2] = best_feature[4][1]
                    else:
                        # The node is a leaf node
                        self.tree[node] = Counter(labels[current_pop])
            nodes = next_nodes
            level += 1
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list

        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:
                    label = list(self.tree[node].keys())[np.argmax(self.tree[node].values())]
                    predictions.append(label)
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # Eample:
        # self.classes_ = {"2", "1"}
        # the reached node for the test data point has {"1":2, "2":1}
        # then the prob for that data point is {"2": 1/3, "1": 2/3}
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)

        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:               
                    # Calculate prediction probabilities for data point arriving at the leaf node.
                    # predictions = list of prob, e.g. prob = {"2": 1/3, "1": 2/3}
                    sum=0
                    for i in self.tree[node]:
                        sum+=self.tree[node][i]
                    prob = {}
                    for i in self.classes_:
                        prob[i]=self.tree[node][i]/sum
                    predictions.append(prob)
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2
        probs = pd.DataFrame(predictions, columns=self.classes_)
        return probs
