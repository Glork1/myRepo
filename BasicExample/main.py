import numpy as np
import pandas as pd
import sklearn.linear_model as sk
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\BasicExample\\Advertising.csv")
table = data.values # numpy array

# Pairwise scatter plot to get side-by-side boxplots
from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha = 0.2, figsize = (10, 10), diagonal = 'hist')


Y = table[:,-1]
X = table[:,1:table.shape[1]-1]

# Simple linear regression: sales explained by TV advertisement budget
X_training = X[:-20,0].reshape(-1, 1)
X_test = X[-20:,0].reshape(-1, 1)

Y_training = Y[:-20]
Y_test = Y[-20:]

reg = sk.LinearRegression()

reg.fit(X_training,Y_training)

Y_predict = reg.predict(X_test)

print("sklearn intercept " + str(reg.intercept_))
print("sklearn coeff " + str(reg.coef_))
print("sklearn R2 score: " + str(r2_score(Y_test,Y_predict)))

plt.figure()
plt.scatter(X_test,Y_test,color='blue')
plt.plot(X_test,Y_predict,color='red') # least square line

RSS = ((Y_test-Y_predict)**2).sum() # residual sum of squares
TSS = ((Y_test-np.mean(Y))**2).sum()
print("RSS: " + str(RSS))
print("TSS: " + str(TSS))
print("my R2: " + str((TSS-RSS)/TSS))
print("RSE: " + str(np.sqrt(RSS/(178))))


# Simple linear regression using statsmodels
import statsmodels.api as sm
X_training = sm.add_constant(X_training) # add a column of 1 to get the intercept
ols = sm.OLS(Y_training, X_training)
ols_result = ols.fit()
#print(ols_result.summary())
# Now you have at your disposition several error estimates, e.g.
print(ols_result.bse) # standard error of the parameter estimate

     
# Multiple linear regression 
Y = table[:,-1]
correlation_matrix = np.corrcoef(X.T)
#X = sm.add_constant(X)


ols = sm.OLS(Y, X)
ols_result = ols.fit()
#print(ols_result.summary())

# Multiple linear regression with interaction terms
Y = table[:,-1].reshape(-1,1)
Xint = np.multiply(X[:,0],X[:,1])
XX = Xint.reshape(-1,1)

X = X[:,:X.shape[1]-1]
X = np.hstack((X,XX))

X = sm.add_constant(X) # add a column of 1 to get the intercept

ols = sm.OLS(Y, X)
ols_result = ols.fit()
print(ols_result.summary())

# Plot of the residuals yi <-> residuals (multiple regression)
residu = ols_result.resid

plt.figure()
plt.scatter(ols_result.fittedvalues,residu,color="green")


# Better to plot the studentized residuals
plt.figure()
stud_results = (ols_result.outlier_test())[:,0]
plt.scatter(ols_result.fittedvalues,stud_results,color="red")



# VIF : if greater than 5 or 10, then problem of collinearity between the variables. 
# So, drop it because the info that this variable gives is redundant 
# OR combine all the collinear variable into a single one
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X[:,1:], i) for i in range((X[:,1:]).shape[1])]






