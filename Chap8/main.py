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

############################################

'''
Sales of Child Car Seats: a simulated data set containing sales of child car seats at 400 different stores. 

Sales

    Unit sales (in thousands) at each location
CompPrice

    Price charged by competitor at each location
Income

    Community income level (in thousands of dollars)
Advertising

    Local advertising budget for company at each location (in thousands of dollars)
Population

    Population size in region (in thousands)
Price

    Price company charges for car seats at each site
ShelveLoc

    A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
Age

    Average age of the local population
Education

    Education level at each location
Urban

    A factor with levels No and Yes to indicate whether the store is in an urban or rural location
US

    A factor with levels No and Yes to indicate whether the store is in the US or not

'''




import pandas as pd

# Loading the Wage dataset
df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap8\\Carseats.csv",header=0, na_values='NA')
df = df.dropna().reset_index(drop=True)
df = df.drop('Unnamed: 0', axis=1)

# Get the headers of the data
# 1st solution
liste_df = list(df)
print(liste_df)

#2nd solution
liste_df_bis = df.columns.tolist()

# Pull a sample
print(df.head())

# Summary statistics
pd.set_option('display.max_columns', None) # to inspect all the columns
pd.set_option('display.float_format', lambda x: '%.2f' % x) # to suppress scientific notation
print(df.describe())
df.hist(figsize=(6,6))

# Pairwise scatter plot to get side-by-side boxplots
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha = 0.2, figsize = (10, 10), diagonal = 'hist')


# Pre-treatment: replace qualitative variables by quantitative ones

# 1st technique
df['High'] = df.Sales.map(lambda x: 1 if x>8 else 0)
df.Urban = df.Urban.map({'No':0, 'Yes':1})

# 2nd technique
#df.Urban.replace(('No', 'Yes'), (0, 1), inplace=True)
df.US.replace(('Yes','No'),(1,0),inplace=True)

'''
Encode the object as an enumerated type or categorical variable.

This method is useful for obtaining a numeric representation of an array when 
all that matters is identifying distinct values. factorize is available as both a top-level function pandas.factorize(),
and as a method Series.factorize() and Index.factorize().
'''


df.ShelveLoc = pd.factorize(df.ShelveLoc)[0] # before bad good, medium in this column. ow 0,1,2
df.info() # n=400, p=12 (including a column "unnamed")

# Pull a sample after the pre-treatment
print(df.head(5))

''''

Decision Tree

Goal here:
    
fit a classification tree in order to predict High with the rest of the Carseats data (except High and Sales itself of course)
'''

X = df.drop(['Sales', 'High'], axis=1)
Y = df.High

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

'''
DecisionTreeClassifier(
criterion=’gini’,  function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
splitter=’best’, The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
max_depth=None, The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
min_samples_split=2, The minimum number of samples required to split an internal node
min_samples_leaf=1, 
min_weight_fraction_leaf=0.0, 
max_features=None, The number of features to consider when looking for the best split:If int, then consider max_features features at each split.If float, then max_features is a percentage.If “sqrt”, then max_features=sqrt(n_features). 
random_state=None, 
max_leaf_nodes=None, 
min_impurity_decrease=0.0, A node will be split if this split induces a decrease of the impurity greater than or equal to this value
min_impurity_split=None, Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
class_weight=None, 
presort=False)
'''

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X, Y)

from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
print(classification_report(Y, clf.predict(X)))

'''
             precision    recall  f1-score   support

          0       0.89      0.99      0.94       236
          1       0.98      0.82      0.89       164

avg / total       0.93      0.92      0.92       400
'''

'''
Example of a classification report:

from sklearn.metrics import classification_report
>>> y_true = [0, 1, 2, 2, 2]
>>> y_pred = [0, 0, 2, 2, 1]
>>> target_names = ['class 0', 'class 1', 'class 2']
>>> print(classification_report(y_true, y_pred, target_names=target_names))
             precision    recall  f1-score   support

          0       0.89      0.99      0.93       236
          1       0.98      0.82      0.89       164

avg / total       0.92      0.92      0.92       400


Suppose a computer program for recognizing dogs in photographs identifies 8 dogs in a picture containing 12 dogs and some cats. 
Of the 8 identified as dogs, 5 actually are dogs (true positives), while the rest are cats (false positives). 
The program's precision is 5/8 while its recall is 5/12.

The F1 Score is the 2*((precision*recall)/(precision+recall)). It is also called the F Score or the F Measure. 
Put another way, the F1 score conveys the balance between the precision and the recall. 
'''


graph3, = print_tree(clf, features=X.columns, class_names=['No', 'Yes'])
my_image = Image(graph3.create_png())
from IPython.display import display
display(my_image)

clf.fit(X_train, Y_train)
pred = clf.predict(X_test)

cm = pd.DataFrame(confusion_matrix(Y_test, pred).T, index=['No', 'Yes'], columns=['No', 'Yes'])
cm.index.name = 'Predicted'
cm.columns.name = 'True'
print(cm)

'''

True        No  Yes
Predicted          
No         100   31
Yes         18   51

'''

# Precision of the model using test data is 74%
print(classification_report(Y_test, pred))

'''
             precision    recall  f1-score   support

          0       0.76      0.85      0.80       118
          1       0.74      0.62      0.68        82

avg / total       0.75      0.76      0.75       200

'''

'''

Regression Tree

'''

boston_df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap8\\Boston.csv",header=0, na_values='NA')
boston_df = boston_df.dropna().reset_index(drop=True)



'''

Goal: predict medv: median value of owner-occupied homes in \$1000s

using everything else
    
'''
X_b = boston_df.drop(['medv','Unnamed: 0'], axis=1)
Y_b = boston_df.medv

X_train_b, X_test_b, Y_train_b, Y_test_b = train_test_split(X_b, Y_b, test_size=0.5, random_state=0)

# Pruning not supported. Choosing max depth 3)
regr2 = DecisionTreeRegressor(max_depth=3)
regr2.fit(X_train_b, Y_train_b)
pred_b = regr2.predict(X_test_b)

graph_b, = print_tree(regr2, features=X_b.columns)
my_image_b = Image(graph_b.create_png())
display(my_image_b)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(pred_b, Y_test_b, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes) # pour tracer la ligne droite en pointille # "gca" for get current axis
plt.xlabel('pred')
plt.ylabel('y_test') # should be on the line if match


print(mean_squared_error(Y_test_b, pred_b)) # 26.023230850097445

     
'''

Bagging and Random Forests

'''
# There are 13 features in the dataset and 206 observations
print(X_b.shape) # (506,13)

     
# Bagging: using all features
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor

regr1 = RandomForestRegressor(max_features=13, random_state=1)
regr1.fit(X_train_b, Y_train_b)

pred_b = regr1.predict(X_test_b)

plt.figure()
plt.scatter(pred_b, Y_test_b, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print(mean_squared_error(Y_test_b, pred_b)) #18.301366007905138 We see a very good improvement

# Random forests: using 6 features (if m = 13 at each step, then it's the bagging from previous example
regr2 = RandomForestRegressor(max_features=6, random_state=1)
regr2.fit(X_train_b, Y_train_b)
pred_b = regr2.predict(X_test_b)

plt.figure()
plt.scatter(pred_b, Y_test_b, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print(mean_squared_error(Y_test_b, pred_b))#16.469374703557314

plt.figure()
Importance = pd.DataFrame({'Importance':regr2.feature_importances_*100}, index=X_b.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
       
       
       
'''

Boosting

'''

regr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=1)
regr.fit(X_train_b, Y_train_b)

feature_importance = regr.feature_importances_*100
rel_imp = pd.Series(feature_importance, index=X_b.columns).sort_values(inplace=False)
print(rel_imp)
rel_imp.T.plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
       
print(mean_squared_error(Y_test_b, regr.predict(X_test_b))) # 15.529710264059759

     
'''

lstat and rm are by far the most importanrt variables      
     
'''