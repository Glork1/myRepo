import numpy as np
import pandas as pd

# Preparing the data
df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\CrossValidation\\Auto.csv", header=0, na_values='?')
df = df.dropna().reset_index(drop=True) # drop the observation with NA values and reindex the obs from 0
df = df[['mpg','cylinders','displacement','horsepower']]

# Setting the seed
np.random.seed(1)

# Summary statistics
pd.set_option('display.max_columns', None) # to inspect all the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # to suppress scientific notation
print(df.describe())
df.hist(figsize=(10,10))

# RESAMPLING METHOD

# CROSS-VALIDATION

### VALIDATION SET APPROACH ###

### 1st technique
# Train Test Split: 
# we split the set of observations into two halves, by selecting a random subset of 196 obs out of 392

import statsmodels.api as sm
X = df.iloc[:,3].values # horsepower
X = sm.add_constant(X)
Y = df.iloc[:,0].values # mpg

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=196, shuffle=True)

# Regression
ols = sm.OLS(Y_train, X_train)
ols_result = ols.fit()
print(ols_result.summary())

### 2nd technique
# + only use statsmodels (no sklearn)
# + add automatically the column of 1 for the intercept
# - implement its own valdiation set procedure

# Train Test Split: 
# we split the set of observations into two halves, by selecting a random subset of 196 obs out of 392
train = np.random.choice(df.shape[0], 196, replace=False)

# below:
# range(3) == [0, 1, 2]
# array of boolean. i-th element is True if it is in the train array
select = np.in1d(range(df.shape[0]), train)

import statsmodels.formula.api as smf

df_select= df[select]

# Regression:
# we fit a linear regression using only the observations corresponding to the training set
# the command just below generates a uniform random sample from np.arange(df.shape[0] [=396]) of size 196:

lm = smf.ols ('mpg~horsepower', data = df_select).fit()
print(lm.summary())

# we use the predict function to estimate the response for all  observations
preds = (lm.predict(df).values).reshape((392,1))
origi = ((df['mpg']).values).reshape((392,1))
square_error = ((origi - preds)**2)

print('--------Test Error for 1st order i.e.: MSE for the 196 obs  of the validation set--------')

# and we use the mean function to calcultate the MSE of the observations in the validation set
# ~ is the complementary operator
print(np.mean(square_error[~select])) # 26.74

# Regression with polynomial regression
lm2 = smf.ols ('mpg~horsepower + I(horsepower ** 2.0)', data = df[select]).fit()
print(lm2.summary())

preds = lm2.predict(df)
square_error = (df['mpg'] - preds)**2
print('--------Test Error for 2nd order--------')
print(square_error[~select].mean()) #21.70

lm3 = smf.ols ('mpg~horsepower + I(horsepower ** 2.0) + I(horsepower ** 3.0)', data = df[select]).fit()
print(lm3.summary())
preds = lm3.predict(df)
square_error = (df['mpg'] - preds)**2
print('--------Test Error for 3rd order--------')
print(np.mean(square_error[~select])) #21.69, slight increase so cubic part is useless, and also p-value fairly big

# If we use a different seed, the validaiton set error rate will be different
# These results are consistent with our previous findings: 
# a model that predicts mpg using a quadratic function of horsepower performs better 
# than a model that involves only a linear function of horsepower, 
# and there is little evidence in favor of a model that uses a cubic function of horsepower.


### LEAVE-ONE-OUT CROSS VALIDATION ###

# OLS fit
ols_fit = smf.ols ('mpg~horsepower', data = df).fit()
print(ols_fit.params)

# GLM fit (G for genralized, gives the same results, compared to OLS)
glm_fit = sm.GLM.from_formula('mpg~horsepower', data = df).fit()
print(glm_fit.params)

# Trying CV in Python is not as easy as that in R. It will require some manual coding.
# To use some of implemented function in Python, we use Sklearn for linear model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

x = pd.DataFrame(df.horsepower)
y = df.mpg

model = LinearRegression()
model.fit(x, y)

# gives again the same results
print(model.intercept_)
print(model.coef_)

#k_fold = KFold(n_splits=x.shape[0]) # loo use folds equal to # of observations
#test = cross_val_score(model, x, y, cv=k_fold,  scoring = 'neg_mean_squared_error', n_jobs=-1)
#print(np.mean(-test)) #39.93

# For higher order polynomial fit, we use pipline tool. 
# Below shows how to fit an order 1 to 5 polynomial data and show the loo results
#A = []
#for porder in range(1, 6):
#    model = Pipeline([('poly', PolynomialFeatures(degree=porder)), ('linear', LinearRegression())])
#    k_fold = KFold(n_splits=x.shape[0]) # loo use folds equal to # of observations
#    test = cross_val_score(model, x, y, cv=k_fold,  scoring = 'neg_mean_squared_error', n_jobs=-1)
#    A.append(np.mean(-test))
#    
#print(A)
# Observation: we can see a sharp drop in the estimated test MSE between the linear and quadratic fits,
# but then  no clear improvement from using higher-order polynomials


### K-FOLD CROSS VALIDATION
# exactly the same as LOO with different n_splits parameter setup. 
# The computation time is much shorter than that of LOOCV

#p.random.seed(2)
# = []
#or porder in range(1, 11):
#   model = Pipeline([('poly', PolynomialFeatures(degree=porder)), ('linear', LinearRegression())])
#   k_fold = KFold(n_splits=10) 
#   test = cross_val_score(model, x, y, cv = k_fold,  scoring = 'neg_mean_squared_error', n_jobs = -1)
#   A.append(np.mean(-test))
#   
#print(A)

# BOOTSTRAP

# means sampling with replacement. 
# To eliminate the effect of sample size, 
# the norm practice is to sample the same size as original dataset with replacement.

portfolio = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\CrossValidation\\Portfolio.csv", header=0)
portfolio = portfolio[['X','Y']]

# To illustrate the use of the bootstrap on this data, 
# we must first create a function, alpha_fn(), 
# which takes as input the (X, Y) data as well as a vector indicating which observations should be used 
# to estimate alpha

def alpha_fn(data,index):
    X = portfolio.X[index]
    Y = portfolio.Y[index]
    return (np.var(Y) - np.cov(X,Y)[0,1])/(np.var(X) + np.var(Y) - 2 * np.cov(X, Y)[0,1])

#true_alpha = alpha_fn(portfolio, range(0, 100))

# Generate one set of random index with 100 elements. 
# The array has been sorted to show there are repeat elements.

ex = np.sort(np.random.choice(range(0,100),size=100,replace=True))
print(alpha_fn(portfolio,ex)) #0.58

def boot_python(data, input_fun, iteration):
    n = portfolio.shape[0]
    idx = np.random.randint(0, n, (iteration, n))
    stat = np.zeros(iteration)
    for i in range(len(idx)):
        stat[i] = input_fun(data, idx[i])
    return {'Mean': np.mean(stat), 'STD': np.std(stat)}

alpha_boost_python = boot_python(portfolio, alpha_fn, 1000)