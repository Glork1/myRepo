''' MNIST Classifier using Random Forest''' 
import numpy as np
np.random.seed(0)

import pandas as pd

# Loading the train data set: each image is just a vector of pixel
df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap8_mnist\\train.csv",header=0, na_values='NA')
tab = df.values
X = tab[:,1:785] # or df.iloc[:,1:]
Y = tab[:,0] # or df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42) # to ESTIMATE the test error only

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1, n_estimators=100)

'''

n_estimators : integer, optional (default=10) : The number of trees in the forest
n_jobs : int or None, optional (default=None) : The number of jobs to run in parallel for both fit and predict

'''
clf.fit(X_train,Y_train)

import matplotlib.pyplot as plt


def plotNum(ind):
    plt.imshow(np.reshape(np.array(df.iloc[ind,1:]), (28, 28)), cmap="gray")
    
plt.figure()
for ii in range(1,17):
    plt.subplot(4,4,ii)
    plotNum(ii-1)
    

print(clf.score(X_test, Y_test)) # 93% !

Y_pred = clf.predict(X) # predict the whole thing, to avoid problem of indices when checking

plt.figure()
plotNum(34)
