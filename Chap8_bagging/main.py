# -*- coding: utf-8 -*-
"""
The mean is a coordinate in N-dimensional space, which represents the location where samples are most likely to be generated. 
This is analogous to the peak of the bell curve for the one-dimensional or univariate normal distribution.

Covariance indicates the level to which two variables vary together. From the multivariate normal distribution, 
we draw N-dimensional samples, X = [x_1, x_2, ... x_N]. The covariance matrix element C_{ij} is the covariance of x_i and x_j. 
The element C_{ii} is the variance of x_i (i.e. its “spread”).
"""


n = 2030 # train on 30 and test  is 2000 size
p = 5
number_class = 2
rho = 0.5

import numpy as np

mean = [0]*p
cov = np.array([[1,rho,rho,rho,rho],[rho,1,rho,rho,rho],[rho,rho,1,rho,rho],[rho,rho,rho,1,rho],[rho,rho,rho,rho,1]])

x1,x2,x3,x4,x5 = np.random.multivariate_normal(mean, cov, n).T
X = (np.vstack((x1,x2,x3,x4,x5))).T

    
import matplotlib.pyplot as plt
plt.plot(x1, x2, 'x')

'''
Response is generated as follow:
P[Y=1|x1<=0.5]=0.2
P[Y=1|x1>0.5]=0.8
'''
y = np.zeros(n)
for i in range(n):
    s = np.random.rand(1,1)
    if (x1[i]>0.5):
        if (s>0.2):
            y[i] = 1
        else:
            y[i] = 0
    else:
        if (s<=0.2):
            y[i] = 1
        else:
            y[i] = 0
            

''' 
Below is taken from Chapter 8
'''
# This function creates images of tree models using pydot
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO  

def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names
    
    dot_data = StringIO() #read and write string as files
    
    '''
    Export a decision tree in DOT format: export_graphviz
    This function generates a GraphViz representation of the decision tree, which is then written into out_file. 
    Once exported, graphical renderings can be generated using
    '''
    
    '''
    filled : bool, optional (default=False)
    When set to True, paint nodes to indicate majority class for classification, extremity of values for regression, or purity of node for multi-output.
    '''
    
    '''
    estimator is the decision tree classifier
    '''
    
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    
    '''
    Load graph as defined by data in DOT format.
    The data is assumed to be in DOT format. It will be parsed and a Dot class will be returned, representing the graph.
    '''
    
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return(graph)

''' 
Fitting a classification tree
'''

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
clf = DecisionTreeClassifier(max_depth=6)

X_train = X[:30,:]
y_train = y[:30]

X_test = X[30:,:]
y_test = y[30:]



clf.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
print(classification_report(y, clf.predict(X)))

'''
             precision    recall  f1-score   support

        0.0       1.00      1.00      1.00        17
        1.0       1.00      1.00      1.00        13

avg / total       1.00      1.00      1.00        30

'''

graph3, = print_tree(clf, features=['x1','x2','x3','x4','x5'], class_names=['No', 'Yes'])
my_image = Image(graph3.create_png())
from IPython.display import display
display(my_image)

y_pred = clf.predict(X_test)

''' Test error rate now '''
import pandas as pd
cm = pd.DataFrame(confusion_matrix(y_test, y_pred).T, index=['No', 'Yes'], columns=['No', 'Yes'])
cm.index.name = 'Predicted'
cm.columns.name = 'True'
print(cm)

'''
True        No  Yes
Predicted          
No         899  524
Yes        363  214

ie: (899+214)/total = 55%

Quite bad ! But ok cf ESL

Let's do bagging to improve accuracy

'''

# Bagging: using all features
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor

B = 1 # number of bagging
regr1 = BaggingClassifier(n_estimators = B, max_features=5, random_state=1)
regr1.fit(X_train, y_train)

pred_b = regr1.predict(X_test)

cm_b = pd.DataFrame(confusion_matrix(y_test, pred_b).T, index=['No', 'Yes'], columns=['No', 'Yes'])
print(cm_b)


'''
Test error rate 
B = 1 : 53 % 
B = 2 : very fast decrease
B = 10 : 65% 

After it fluctuates ...

The Bayes error rate is 0.2.

'''









