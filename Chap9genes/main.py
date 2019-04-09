''' Application do Gene Expression Data Set '''

import pandas as pd

# Loading the Wage dataset

# (63,2308)
X_train = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap9genes\\Khan_xtrain.csv").drop('Unnamed: 0', axis=1)

# (20,2308)
X_test = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap9genes\\Khan_xtest.csv").drop('Unnamed: 0', axis=1)

# (63,)
Y_train = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap9genes\\Khan_ytrain.csv").drop('Unnamed: 0', axis=1).values.reshape(-1,1)#as_matrix().ravel()

# (20,)
Y_test = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap9genes\\Khan_ytest.csv").drop('Unnamed: 0', axis=1).values.reshape(-1,1)#.as_matrix().ravel()

# Summary statistics
#print(pd.Series(Y_train).value_counts(sort=False)) # don't work
'''
1     8
2    23
3    12
4    20
'''
#print(pd.Series(Y_test).value_counts(sort=False)) # don't work
'''
1    3
2    6
3    6
4    5
'''


#print(X_train.describe()) # Slow

''' Support Vector Classifier '''

from sklearn import svm

svc = svm.SVC(C= 30, kernel='linear')
svc.fit(X_train,Y_train)

Y_pred = svc.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, Y_pred)
cm_df = pd.DataFrame(cm.T, index=svc.classes_, columns=svc.classes_)
cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
#or... print(pd.DataFrame(confusion_matrix(Y_train, Y_pred),index=svc.classes_, columns=svc.classes_))
print(cm_df)

'''

True       1   2   3   4
Predicted               
1          8   0   0   0
2          0  23   0   0
3          0   0  12   0
4          0   0   0  20

No training error
No surpising, because the large number of variables relative to the number of observations
implied that it is easy to find hyperplanes that fully separate the classes

However here, it's training performance that we are evaluatingm not test performance (see below)
'''

Y_pred_test = svc.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred_test)
cm_df = pd.DataFrame(cm.T, index=svc.classes_, columns=svc.classes_)
cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
#or... print(pd.DataFrame(confusion_matrix(Y_test, Y_pred_test),index=svc.classes_, columns=svc.classes_))
print(cm_df)

'''
True       1  2  3  4
Predicted            
1          3  0  0  0
2          0  6  2  0
3          0  0  4  0
4          0  0  0  5

Testing error rate: 18/20 = 90%
'''