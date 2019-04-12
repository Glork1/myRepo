import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Classif\\Smarket.csv")#,delimiter=',')

# Logistic regression

# Dimensions

print(df.shape) # (1250,9)

# Dataframe manipulation

pd.set_option('display.max_columns', None) # to inspect all the columns
pd.set_option('display.float_format', lambda x: '%.5f' % x) # to suppress scientific notation

print(df.columns) # list all the columns (in a dataframe) 
list_columns = df.columns.tolist() # one might want to not include all the variables in the summary statistic table
#df = df[['Lag1','Lag2','Lag3','Lag4','Lag5','Volume','Today','Direction']] # now we limit the dataframe to just the columns we want

# Summary statistics

print(df.describe()) # generate a summary for each column. 25% etc return the 25% percentile. df.describe('percentiles=None' to remove it)
df.hist()# plot histogram for each of the varuable
# we see the correlations are close to zero between lag variables and today today's return. 
# in other words, there is little correlation  between ... and ...
# only substantial correlation is between Year and Volume
correlation_matrix = np.corrcoef((df[['Year','Lag1','Lag2','Lag3','Lag4','Lag5','Volume','Today']]).T) 

# let's plot the volume as a function of time

import matplotlib.pyplot as plt

# we indeed see that the volume is increasing when Year is increasing
fig = plt.figure(1,figsize=(15, 5))
plt.plot(df[['Volume']])

# Logistic regression
# Aim: fit a logistic regression model in order to predit Direction using Lag1..Lag5 and Volume

import statsmodels.api as sm

df.Direction.replace(('Up', 'Down'), (1, 0), inplace=True) # replace otherwise error...
df = sm.add_constant(df) # add a column of 1 to get the intercept
logit = sm.Logit(df['Direction'], df[['const','Lag1','Lag2','Lag3','Lag4','Lag5','Volume']])
logit_result = logit.fit()
print(logit_result.summary())

# We see that the smallest p-value is 0.145, associated with Lag1.
# The coeff is <0, meaning that an increase in return of the previous day will decrease the probability of being Up on today's date // less likely to go up today
# However, this p-value is relatively large, so no clear evidence of a real association between Lag1 and Direction


# doing some "predicted" (on the training dataset) probabilities of Up/Down
prob = logit_result.predict(df[['const','Lag1','Lag2','Lag3','Lag4','Lag5','Volume']])
prob= prob.values # convert a series to numpy array
pred = ["Down"]*1250 # creates a list of 1250 "Down" elements
pred = np.where(prob>0.5,"Up","Down") # actually all above is useless since one can use the pred_table function
print(logit_result.pred_table()) # in order tp get the confusion matrix

# Confusion matrix

#             Direction
#             UP  DOWN
# Pred UP   [[145 457]
#      DOWN  [141 507]]

# Meaning (145+507)/1250 = 52.2% have been correctly predicted. That is, just a bit better than random guessing.
# Also, the testing error rate will be much more worse (traning error rate is 100-52.2 = 47.8%)

# Training error rate
##new_directions = df.Direction.replace((1,0), ('Up', 'Down'), inplace=True) #  <- weird

##frac_good_prediction = np.mean(np.sum(pred==new_directions)) # <- weird Training error rate is ~51% worse than naive approach

# 1st solution: because all p-value are not that small, only consider Lag, Lag1 as predictors
# useless predictors (those who don't have any relationship) cause an increase in variance without a corresponding decrease in bias (because they are useless)
# after doing that, results seem to be better : 56% of daily movements correctly predicted
# but naive approach "increase every day" wil also be correct 56% of the time !
# But, confusion matrix shows that:
# on days when logistic regression predicts an increase in the market, it has a 58% accuracy rate
# Possible trading strategy: buying on days when the model predicts an increasing market
# But: possible due to random chance 



# LDA

4
5
	
# Applying Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components = 2)

# Importing the dataset
dataset = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Classif\\Smarket.csv")#,delimiter=',')

X = dataset.iloc[:,0:dataset.shape[1]-1].values
Y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
# Split arrays or matrices into random train and test subsets
# Signature: train_test_split(X, Y, test_size = 0.1 if float:proportion of the dataset to include in the test split e.g. 10%. If int: absloute, random_state = 0) 
# random_state: seed used by the random number generator
# schuffle: wether or not to shuffle the data before splitting
# Example: list(y) : [0, 1, 2, 3, 4]
# train_test_split(y, shuffle=False): [[0, 1, 2], [3, 4]]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 998, random_state = 0, shuffle=False) # so here: ramdom_state is useless. 

# Applying Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)
X_train = lda.fit_transform(X_train, Y_train) # cf wine classifier: basically optimisation (LDA dimensionality reduction)
X_test = lda.transform(X_test)

#LDA_frac_good_prediction = np.mean(np.sum(Y_pred_up_down==Y_test)) <- finish TODO

# QDA

# ... tofo

# K-nearest neighbors

# Preprocessing
knn_dataset = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Classif\\Smarket.csv")#,delimiter=',')

X = knn_dataset.iloc[:,0:knn_dataset.shape[1]-1].values
Y = knn_dataset.iloc[:,-1].values

# Train Test Split
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=998, shuffle=False) #0.20)

# Feature Scaling

# Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated
# Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 
# For example, the majority of classifiers calculate the distance between two points by the Euclidean distance. 
# If one of the features has a broad range of values, the distance will be governed by this particular feature. 
# Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
# The gradient descent algorithm (which is used in neural network training and other machine learning algorithms) also converges faster with normalized features.

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

# Training and Predictions

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))  

# Confusion matrix

#             Direction
#             UP  DOWN
# Pred UP   [[402 70]
#      DOWN  [177 349]]

# meaning (402+349)/1250 = 60%

          

# Comparing Error Rate with the K Value
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  