import numpy as np

np.random.seed(9)

# Generating random data: 20 observations of 2 features and divide into two classes.
X = np.random.rand(20,2) # features
Y = np.repeat([1,-1], 10) # response (the class)

X[Y == -1] = X[Y == -1] + 0.1 # we jus shift the observation by adding elementwise +1 on x-axis ans y-axis

import matplotlib.pyplot as plt
 
'''
s: the marker size in points**2.
c: the color will be based on the category
'''

plt.scatter(X[:,0], X[:,1], s=70, c=Y, cmap=plt.cm.Paired)

plt.xlabel('X1')
plt.ylabel('X2');
plt.ylabel('These will be our training data');

''' Support Vector Classifier '''

from sklearn import svm

# Function to plot a classifier with support vectors

def plot_svc(svc, X, y,title, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.show()
    print('Number of support vectors: ', svc.support_.size)



svc = svm.SVC(C= 30, kernel='linear')

'''C : float, optional (default=1.0): p[nalty parameter C of the error term.'''
svc.fit(X,Y)
plot_svc(svc, X, Y,'Using SVC with C=30 on training data')

'''
When using a smaller cost parameter (C=0.1) the margin is wider, resulting in more support vectors.
Be careful: different from the definition of 'C' used in the Support Vector Classifier Book's Section.
The margin is also wider, when using a smaller C, since the support vectors are mroe far to the decision boundary.

'''

svc2 = svm.SVC(C=0.1, kernel='linear')
svc2.fit(X, Y)
plot_svc(svc2, X, Y,'Using SVC with C=0.1 on training data')



# Select the optimal C parameter by cross-validation
from sklearn.model_selection import train_test_split, GridSearchCV

tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
clf = GridSearchCV(svm.SVC(kernel='linear'), tuned_parameters, cv=10, scoring='accuracy', return_train_score=True)
clf.fit(X, Y)
print(clf.cv_results_)

print(clf.best_params_) # {'C': 0.001} is the best param

# Now using this optimized "cost" parameter on test data
np.random.seed(2) # we use other seed otherwise...
X_test = np.random.rand(20,2)
Y_test = np.repeat([1,-1],10)
X_test[Y_test==1] = X_test[Y_test==1]+0.1

plt.figure()
plt.title('We generate test data: on the next step, we will use the optimized fitted SVC on these test data ')
plt.scatter(X_test[:,0], X_test[:,1], s=70, c=Y_test, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2');

          
svc_opt = svm.SVC(C=0.001,kernel='linear')
svc_opt.fit(X,Y) # we fit using the previous training observations
Y_pred = svc_opt.predict(X_test)

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import pandas as pd

print(pd.DataFrame(confusion_matrix(Y_test, Y_pred),index=svc.classes_, columns=svc.classes_))

'''
    -1   1
-1   1   9
 1   4   6

Accuracy is then : 7/13 = 53%
'''

svc_opt_2 = svm.SVC(C=0.01,kernel='linear')
svc_opt.fit(X,Y)
Y_pred = svc_opt.predict(X_test)
print(pd.DataFrame(confusion_matrix(Y_test, Y_pred),index=svc.classes_, columns=svc.classes_))

'''
    -1   1
-1   1   9
 1   4   6

Accuracy is SAME : 7/13 = 53%
'''

''' Support Vector Machine '''

# Let's first generate training and test data

X = np.random.randn(200,2)
Y = np.repeat([1,-1],100)
X[:100] = X[:100] + 2
X[101:150] = X[101:150] -2
Y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=2)
plt.figure()
plt.scatter(X[:,0],X[:,1],s=70, c=Y, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2');

'''
We clearly see that the boundary is non-linear:
a group of blue points is kind of surrounding a group of red points
'''
          
svm_radial_kernel = svm.SVC(C=1.0,kernel='rbf',gamma=1)
svm_radial_kernel.fit(X_train,Y_train)
plot_svc(svm_radial_kernel, X_train, Y_train,'Radial Kernel fit on training observations, with C = 1.0')

svm_radial_kernel_2 = svm.SVC(C=100.0,kernel='rbf',gamma=1)
svm_radial_kernel_2.fit(X_train,Y_train)
plot_svc(svm_radial_kernel_2, X_train, Y_train,'Radial Kernel fit on train obs with an increased C = 100.0, allowing more flexibility')


# Set the parameters by cross-validation
tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],'gamma': [0.5, 1,2,3,4]}]
clf = GridSearchCV(svm.SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy', return_train_score=True)
clf.fit(X_train, Y_train)
print(clf.cv_results_)
print(clf.best_params_) # {'C': 1, 'gamma': 2}

print(pd.DataFrame(confusion_matrix(Y_test, clf.best_estimator_.predict(X_test)),index=svc.classes_, columns=svc.classes_))

'''
    -1   1
-1  70   3
 1  11  16
 
Accuracy is 86/100 = 86% ie 14% of test observations are missclassified

'''



''' Support Vector Nachine with Multiple Classes '''
# Adding a third class of observations
np.random.seed(8)
XX = np.vstack([X, np.random.randn(50,2)])
YY = np.hstack([Y, np.repeat(0,50)])
XX[YY ==0] = XX[YY == 0] + 4

plt.figure()
plt.scatter(XX[:,0], XX[:,1], s=70, c=YY, cmap=plt.cm.prism)
plt.xlabel('XX1')
plt.ylabel('XX2');

svm5 = svm.SVC(C=1, kernel='rbf')
svm5.fit(XX, YY)

plot_svc(svm5, XX, YY,'Performing one-versus-one approach')
print(pd.DataFrame(confusion_matrix(YY, svm5.predict(XX)),index=svm5.classes_, columns=svm5.classes_))

'''
Confusion matrix on training data

     -1   0   1
-1  140   2   8
 0    7  43   0
 1   13   0  37
 
Accuracy on training data is : 220/248 = 89%
'''



