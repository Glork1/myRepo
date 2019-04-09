# Chapter 6: linear model selection and regularization

# To remove the warming  in the window
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd

# Preparing the data (322 observations, 20 columns)
df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap6\\Hitters.csv", header=0, na_values='NA')

#df = df.dropna().reset_index(drop=True) # drop the observation with NA values and reindex the obs from 0

#df = df[['mpg','cylinders','displacement','horsepower']]

# Setting the seed
#np.random.seed(1)


# Get the headers of the data
# 1st solution
liste_df = list(df) # [Unnamed:0, AtBat, Hits,..., Salar,NewLeague]
print(liste_df)

#2nd solution
liste_df2 = df.columns.tolist()

# Pull a sample
print(df.head())

# Summary statistics
pd.set_option('display.max_columns', None) # to inspect all the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # to suppress scientific notation
print(df.describe())
df.hist(figsize=(15,15))

# Count the missing elements: here are 59
print((np.sum(pd.isnull(df['Salary'])))) # note that the count in Salary summary gives the right number

# Drop the observation with NA values and reindex
df = df.dropna().reset_index(drop=True)

# Finalize the preparation of the data
Y = df.Salary  # the response variable: 263 x 1

"""
take care of the features 
1. change category into dummy variables 
2. Choose (n-1) dummy variable into the feature set: n is the unique values of each categorical variable.
"""
# These variables are now 0 or 1 for each one (were string that could only take 2 values before, e.g. for Division W or E, now converted to a numeric value)
dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']]) 
# output are 6 columns: League_A, LEague_N, Division_E, Division_W...
print(dummies.head())

# We replace the 3 variables by the 3 dummy ones (we only take 3 out of the 6 above since complementary)
X_prep = df.drop (['Unnamed: 0', 'Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')
X = pd.concat([X_prep,  dummies[['League_A', 'Division_E', 'NewLeague_A']]], axis=1) # X is now 263 x 19

import statsmodels.formula.api as smf

"""
Since in Python there is no well-defined function for best subset selection, 
we will need to define some functions ourselves.
1. Define a function to run on a subset of feature and extract RSS
2. Select the best model (models) for a fix number of features
"""

def getRSS(Y, X, feature_list):
    lm = smf.OLS (Y, X[list(feature_list)]).fit() #  use the full set for the training. ATTENTION: OLS not ols
    RSS = ((lm.predict(X[list(feature_list)]) - Y) ** 2).sum() # Sum(yi -yi^)^{2}
    return {'Model':lm, "RSS":RSS}

# OLS is the actual model class
# ols from formula.api is just a convenient alias for the method OLS.from_formula 
# that preprocesses the formula information before creating an OLS instance. 
# example
"""
import statsmodels.formula.api as smf
lm = smf.ols ('mpg~horsepower', data = df_select).fit()

BUT

import statsmodels.api as sm
ols = sm.OLS(Y_train, X_train)

"""


# Best subset selection

import itertools

def bestModel(Y, X, K):
    results = []
    for c in itertools.combinations(X.columns, K):
        results.append(getRSS(Y,X,c))
    model_all =  pd.DataFrame(results)
    best_model = model_all.loc[model_all["RSS"].argmin()] # this could be modified to have the top several models
    return best_model

models = pd.DataFrame(columns=["RSS", "Model"])

max_features = 3 # as an example

for i in range(1,(max_features+1)):  # for illustration purpose, I just run for 1 - max_feature features 
    models.loc[i] = bestModel(Y, X, i) # put the model in "i"
    
print(models.loc[2, 'Model'].summary()) # access the model summary of the 2nd model
# this summary confirms that the best two variable model contains the variables Hits and CRBI

""" this show an example to plot the RSS of best models with different number of parameters
bud doesn't say anything if it's a good model on a testing set or not..."""
import matplotlib.pyplot as plt

plt.figure()
plt.plot(models["RSS"])
plt.xlabel('# features')
plt.ylabel('RSS')
plt.show()

# Find the adjusted R^2, use dir() to identify all available attributes
rsquared_adj = models.apply(lambda row: row[1].rsquared_adj, axis=1)

"""
The following graph shows the adj R^2 is still increasing, 
in this case, it is a good idea trying models with more features. 
"""
plt.figure()
plt.plot(rsquared_adj)
plt.xlabel('# features')
plt.ylabel('Adjust R^2')
plt.show()

"""
We can use the previous user defined function 'def getRSS(y, X, feature_list)' to add 
1 feature at a time (start from 0 feature) for forward stepwise selection
or delete 1 feature at a time(start from all the features) for backward stepwise selection. 
"""

# Forward stepwise selection

def forward_select(Y, X, feature_list):
    remaining_predictors = [p for p in X.columns if p not in feature_list]
    results = []
    for p in remaining_predictors:
        results.append(getRSS(Y,X,feature_list+[p]))
    models = pd.DataFrame(results)
    best_model = models.loc[models['RSS'].argmin()]
    return best_model

models2 = pd.DataFrame(columns=["RSS", "Model"])
feature_list = []
for i in range(1,len(X.columns)+1):
    models2.loc[i] = forward_select(Y, X, feature_list)
    feature_list = models2.loc[i]["Model"].model.exog_names

# above: when the model is created with a formula, then the parameter names are stored internally in the data attribute of models, 
# model.data.xnames, and can be accessed through model.exog_names

"""we can compare the results of best subset selection and the forward selection"""
max_feature = 3
print('Best max_feature variable from best subset selection on tranining')
print(models.loc[max_feature, 'Model'].params)
# Output of the best subset selection
#Hits      2.316
#CRBI      0.667
#PutOuts   0.261

print('\n---------------------------------------------')
print('Best max_feature variable from forward selection on tranining')
print(models2.loc[max_feature, 'Model'].params)
# Output of the stepwise forward selection
#Hits      2.316
#CRBI      0.667
#PutOuts   0.261
                 
# Since in this case they select the same variablesm, coefficients should be equal between best subset selection and stepwise forward selection

# Backward stepwise selection

def backward_select(Y,X,features_list):
    results = []
    for combo in itertools.combinations(feature_list, len(feature_list)-1):
        results.append(getRSS(Y, X, combo))
    models = pd.DataFrame(results)
    best_model = models.loc[models['RSS'].argmin()]
    return best_model

"""
The backward selection starts from all the variables of features
"""
models3 = pd.DataFrame(columns=["RSS", "Model"], index = range(1,len(X.columns)))
feature_list = X.columns

while(len(feature_list) > 1):
    models3.loc[len(feature_list)-1] = backward_select(Y, X, feature_list)
    feature_list = models3.loc[len(feature_list)-1]["Model"].model.exog_names

print(models3.loc[max_feature, "Model"].params)

# Output of the stepwise backward selection
#Hits      2.112
#CRuns     0.646
#PutOuts   0.296



# TODO: Choosing Among Models Using the Validation Set Approach and Cross-Validation
# Until now, the procedures return a set of models M0,...,Mp, each one being the best one in its category:
# e.g. for best subset: the model Mi is the best model containing exactly i predictors, among all the models containing i predictors
# and for stepwise fwd selection: the model Mi is the best model containing i predictors, after adding 1 variable to the Mi-1 model, that best minimizes the RSS
# this set of best models is based on a training dataset
# now we need to consider the test MSE, to elect the "true" best model






#  Ridge regression and  Lasso

# Preparing the data (322 observations, 20 columns)

hitters = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap6\\Hitters.csv", header=0, na_values='NA')
hitters = hitters.dropna().reset_index(drop=True)

dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']])

Y = hitters.Salary  # the response variable 
X_prep = df.drop (['Unnamed: 0', 'Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')
X = pd.concat([X_prep,  dummies[['League_A', 'Division_E', 'NewLeague_A']]], axis=1)

#  Ridge regression

"""

Next, we will generate a few candidates lambda(in sklearn, the keyword is alphas) for our Ridge regression. 
In R, alpha is a switch for Ridge and Lasso methods.

"""


from sklearn.preprocessing import scale 

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

alphas = 10**np.linspace(10,-2,100)

"""
Associated with each value of alpha is a vector of ridge regression coefficients, stored in a matrix that can be accessed by coeffs. 
In this case, it is a 19×100, 19 is the dimension of the features + (intercept needs to call separately) and 100 is the len of the alphas. 
The result is a numpy series with len 100 and len(coffes[0]) is 19. In this specific implementation, the default is no intercept.

"""

ridge = Ridge(fit_intercept=True, normalize=True) # IMPORTANT: we normalize the coefficients !
coeffs = []
intercepts = []

# We now just store the coeff (and intercept) resulting for the fit of a Ridge regression for different values of Lambda
# We see that lamnda = 0 yields no shrinkage, and for lamnda->inf, all the coeff converge towards zero
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, Y)
    coeffs.append(ridge.coef_)
    intercepts.append(ridge.intercept_)

print(len(coeffs)) #  100 different values of the tuning parameter
print(len(coeffs[0])) # 19. For each value of the tuning parameter, Ridge regression gives 19 coeff (EXCLUDING the intercept)
print(len(intercepts)) # 100. These are the the different values of the intercept for the Ridge regression
print(intercepts[0]) # 535,92. Try run print len(intercepts[0])

ax = plt.gca() # get the current Axes instance on the current figure
ax.plot(alphas, coeffs)
ax.set_xscale('log') # try without this line
plt.axis('tight') # changes x and y axis limits such that all data is shown
plt.xlabel('alpha') # x title
plt.ylabel('weights') # y title
plt.show()

# Train Test Split

"""
We now split the samples into a training set and a test set in order to estimate the test error of ridge regression and the lasso.
Python provides a built-in function to produce training and test data set.

"""
# We now want to know what is the optimal value of lambda. 
# We evaluate the test MSE (using the validation set procedure), to know what is the best value of lambda.
from sklearn import cross_validation
# Split arrays or matrices into random train and test subsets
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.66)


"""

1st technique:
from sklearn import cross_validation
list(b) = [0, 1, 2, 3, 4] then cross_validation on b
b_train = [2, 0, 3] and b_test = [1, 4]
The option to shuffle or not is not available, but it shuffles by default

2nd technique:
We could have done :
    
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=998, shuffle=False)

In this case, the shuffle option is available
shuffle = True is the default option
shuffle = False yields no shuffle as below:
    
train_test_split(y, shuffle=False)
[[0, 1, 2], [3, 4]]

"""
# Fit a ridge regression on the training data
ridge4 = Ridge(fit_intercept=True, normalize=True, alpha=4)
ridge4.fit(X_train, Y_train)

# Use this model to predict the test data
pred4 = ridge.predict(X_test)

# Print coefficients
print(pd.Series(ridge4.coef_, index=X.columns)) 
print(mean_squared_error(Y_test, pred4)) # this prints the MSE_training = 102966.32840373008

"""

To select best alpha, we will use cross validation. And as standard, we will report test set performance as the final performance metric

"""

ridgecv =  RidgeCV(alphas, scoring='mean_squared_error', normalize = True)
ridgecv.fit(X_train, Y_train)
print(ridgecv.alpha_) #0.49770235643321137


""""

We refit the model using the optimized tuning parameter

"""

ridge_best_fit = Ridge(fit_intercept=True, normalize=True, alpha=ridgecv.alpha_)
ridge_best_fit.fit(X_train, Y_train)

Y_best_pred = ridge_best_fit.predict(X_test)
print(mean_squared_error(Y_test, Y_best_pred)) # this prints the MSE_test = 122337.39431578683. As usual, almost always bigger than MSE_training


"""

If we exam the values of the coefficients, most of them are tiny, but none of them is zero.¶

"""

mySeries_best_Ridge = pd.Series(ridge_best_fit.coef_, index=X.columns)

# Lasso

lasso= Lasso(normalize=True, max_iter=1e5) 
coeffs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, Y_train)
    coeffs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas, coeffs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

lassocv = LassoCV(alphas=None, cv=10, max_iter=1e5, normalize=True)
lassocv.fit(X_train, Y_train)

print(lassocv.alpha_) #3.55
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, Y_train)
print(mean_squared_error(Y_test, lasso.predict(X_test))) # MSE_test = 123198.85337089012 for this optimized Lasso, bigger than the Ridge regression

# Some of the coefficients should reduce to exactly zero
mySeries_best_Lasso = pd.Series(lasso.coef_, index=X.columns)
# Hits : 2.46
# Walks: 2.44
# CHmRun: 0.441
# CRBI: 0.446
# PutOut: 0.418
# Division_E: 123
# other coeffficients are exactly zero

# PCR and PLS Regression

# PCR
hitters = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap6\\Hitters.csv", header=0, na_values='NA')
hitters = hitters.dropna().reset_index(drop=True)

dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']])

Y = hitters.Salary  # the response variable 
X_prep = df.drop (['Unnamed: 0', 'Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')
X = pd.concat([X_prep,  dummies[['League_A', 'Division_E', 'NewLeague_A']]], axis=1)

from sklearn.decomposition import PCA
pca = PCA()

# Scale and transform data to get Principal Components
X_reduced = pca.fit_transform(scale(X)) # Standardize a dataset along any axis. Now we get Z1,...,ZM the M principal components

cum_array = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.figure()
plt.plot(cum_array)
plt.xlabel('# axes')
plt.ylabel('cumulative var')
plt.show()

# Seems like the first two components indeed explain most of the variance in the WHOLE dataset.

# Now we need a procedure to select the right number of principal components for the TESTING SET
# 10-fold CV, with shuffle

n = len(X_reduced)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)

from sklearn.linear_model import LinearRegression

regr = LinearRegression() # fit_intercept : boolean, optional, default True 
# TODO TO CONFIRM
mse = []

# Do one CV to get MSE for just the intercept (no principal components in regression)
score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)), Y.ravel(), cv=kf_10, scoring='mean_squared_error').mean()

"""

regr : estimator object implementing ‘fit’
np.ones((n,1)): The data to fit
Y.ravel(): The target variable to try to predict in the case of supervised learning. x = np.array([[1, 2, 3], [4, 5, 6]]) x.ravel() gives [1 2 3 4 5 6]

cross_val_score returns an array of scores of the estimator for each run of the cross validation
array of float, shape=(len(list(cv)),) here there will be 10 MSE different MSE scores, so we do the mean of this array
scoring='mean_squared_error' (by default R square adjusted)
The actual MSE is simply the positive version of the number you're getting.

"""

mse.append(score) 

# Do CV for the 16 principle components, adding one component to the regression at the time
for i in np.arange(1,16):
    score = -1*cross_validation.cross_val_score(regr, X_reduced[:,:i], Y.ravel(), cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(score)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot(mse, '-v')
ax2.plot(mse[1:16], '-v')
ax2.set_title('Intercept excluded from plot')

for ax in fig.axes:
    ax.set_xlabel('Number of principal components in regression')
    ax.set_ylabel('MSE')
    #ax.set_xlim((-0.2,5.2))

# The minimum test MSE occurs for M = 6 components

# PLS

from sklearn.cross_decomposition import PLSRegression

mse_pls = []

for i in np.arange(1, 16):
    pls = PLSRegression(n_components=i, scale=False)
    pls.fit(scale(X_reduced),Y)
    score = cross_validation.cross_val_score(pls, X_reduced, Y, cv=kf_10, scoring='mean_squared_error').mean()
    mse_pls.append(-score)

plt.figure()
plt.plot(np.arange(1, 16), np.array(mse_pls), '-v')
plt.xlabel('Number of principal components in PLS regression')
plt.ylabel('MSE')
#plt.xlim((-0.2, 5.2))

# The minimum test MSE occurs for M = 11 components



