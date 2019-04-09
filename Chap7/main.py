import pandas as pd

# Loading the Wage dataset
df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap7\\Wage.csv",header=0, na_values='NA')
df = df.dropna().reset_index(drop=True)

# Get the headers of the data
# 1st solution
liste_df = list(df) # [Unnamed:0, AtBat, Hits,..., Salar,NewLeague]
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

''' Polynomials Regression and Step Functions '''
import statsmodels.formula.api as smf

# 1st look
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df[['age']],df[['wage']])
plt.title('Wage as a function of Age')  
plt.xlabel('age')  
plt.ylabel('wage')  

'''

We clearly see 2 categories of salaries 

'''

# Simple linear regression

X = df[['age']] # X is a (3000,1) Pandas Dataframe
Y = df[['wage']] # Y is a (3000,1) Pandas Dataframe
Y_values = Y.values # transforms a (3000,1) Pandas Dataframe into a (3000,1) numpy float array

lm = smf.ols ('wage~age', data = df).fit()

print(lm.summary())

Y_pred_lin = lm.predict(X) # Y _pred_lin is a (3000,) Pandas Series

plt.plot(X,Y_pred_lin,color='red')

Ypl = Y_pred_lin.values.reshape(-1,1) # .values to transform it into a (3000,) numpy array and then reshape to force it to be a (3000,1) array

MSE_training_linear= ((Ypl-Y_values)**2).sum() #5022216...

# TODO quantify the test MSE  with CV

# Polynomial regression
lm_quad = smf.ols ('wage~age+I(age**2)', data = df).fit()

print(lm_quad.summary())

Y_pred_quad = lm_quad.predict(X)

plt.plot(X,Y_pred_quad,'.k')

Ypq = Y_pred_quad.values.reshape(-1,1)

MSE_training_quad = ((Ypq-Y_values)**2).sum() #4793430...

# TODO quantify the test MSE  with CV

'''

We see that MSE_training_linear>MSE_training_quadratic.
This could be predictedjust by considering the shape of the 2 fits.

Let's see focus now on the degree of the polynomial.
2 methods: p_values of cross_validation

d = 1: p-values for intercept and beta_1 very tiny -> d = 1 is a must
d = 2: p-values for intercept, beta_1, beta_2 very tiny -> d = 2 is a must
d = 3: p-values for intercept, beta_1, beta_2 very tiny, p_beta_3 = 0.002 -> d = 3 is still ok
d = 4: p-values for intercept, beta_1, beta_2, beta_3 tiny, p_beta_4 = 0.051 -> d = 4 is a bit useless. And it also increases the p_values of the other predictors
d = 5: all p-values greater than 0.5...

The F-stat is also decreasing each time we add another predictors...

+ cf collinearity warning of statsmodels
+ zoom out the figure to see the strange behaviors of high degreee polynomials 


'''
lm_cub = smf.ols ('wage~age+I(age**2)+I(age**3)', data = df).fit()
print(lm_cub.summary())
Y_pred_cub= lm_cub.predict(X)
plt.plot(X,Y_pred_cub,'.g')
Ypc = Y_pred_cub.values.reshape(-1,1)
MSE_training_cub = ((Ypc-Y_values)**2).sum() #4777674...

lm_4 = smf.ols ('wage~age+I(age**2)+I(age**3)+I(age**4)', data = df).fit()
print(lm_4.summary())
Y_pred_4= lm_4.predict(X)
plt.plot(X,Y_pred_4,'.y')
Yp4 = Y_pred_4.values.reshape(-1,1)
MSE_training_4 = ((Yp4-Y_values)**2).sum() #4771604...

lm_5 = smf.ols ('wage~age+I(age**2)+I(age**3)+I(age**4)+I(age**5)', data = df).fit()
print(lm_5.summary())
Y_pred_5= lm_5.predict(X)
plt.plot(X,Y_pred_5,'.c')
Yp5 = Y_pred_5.values.reshape(-1,1)
MSE_training_5 = ((Yp5-Y_values)**2).sum() #4771604...

                 
'''

Now, predicting whether an individual earns more than 250K/year.

Be careful cf @:
    
The difference is due to the presence of intercept or not:

- in statsmodels.formula.api, similarly to the R approach, a constant is automatically added to your data and an intercept in fitted

- in statsmodels.api, you have to add a constant yourself (see the documentation here). Try using add_constant from statsmodels.api
  x1 = sm.add_constant(x1)

'''

'''

Below is another techniue. Same as before,
but we do the polynomial regression and the logistic polynomial regression at the same time
'''

# Next we consider the task of predicting whether an individual earns more than $250,000 per year

# We first fit the polynomial regression model using the following commands:
from sklearn.preprocessing import PolynomialFeatures

'''
For example, if an input sample is two dimensional and of the form [a, b], 
the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]
>>> X = np.arange(6).reshape(3, 2)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> poly = PolynomialFeatures(2)
>>> poly.fit_transform(X)
array([[  1.,   0.,   1.,   0.,   0.,   1.],
       [  1.,   2.,   3.,   4.,   6.,   9.],
       [  1.,   4.,   5.,  16.,  20.,  25.]])
>>> poly = PolynomialFeatures(interaction_only=True)
>>> poly.fit_transform(X)
array([[  1.,   0.,   1.,   0.],
       [  1.,   2.,   3.,   6.],
       [  1.,   4.,   5.,  20.]])

'''

X1 = PolynomialFeatures(1).fit_transform(df.age.values.reshape(-1,1))
X2 = PolynomialFeatures(2).fit_transform(df.age.values.reshape(-1,1))
X3 = PolynomialFeatures(3).fit_transform(df.age.values.reshape(-1,1))
X4 = PolynomialFeatures(4).fit_transform(df.age.values.reshape(-1,1))
X5 = PolynomialFeatures(5).fit_transform(df.age.values.reshape(-1,1))

# This syntax fits a linear model, using the PolynomialFeatures() function,
# in order to predict wage using up to a fourth-degree polynomial in age.
# The PolynomialFeatures() command allows us to avoid having to write out a 
# long formula with powers of age. We can then fit our linear model:
import statsmodels.api as sm

fit2 = sm.GLS(df.wage, X4).fit()
print(fit2.summary().tables[1]) # We get the exact same coefficients for a d=4-polynomial regression as the first method above 

# Next we consider the task of predicting whether an individual earns more than $250,000 per year. 
# We proceed much as before, except that first we create the appropriate response vector, 
# and then we fit a logistic model using the GLM() function from statsmodels:

# Create response matrix

# Converts the frame to its Numpy-array representation (and not a numpy matrix)  
# Deprecated since version 0.23.0: Use DataFrame.values() instead.
y = (df.wage > 250).map({False:0, True:1}).as_matrix() 

# Fit logistic model
# TODO understandn it!!!
clf = sm.GLM(y, X4, family=sm.families.Binomial(sm.families.links.logit))
res = clf.fit()

# We now create a grid of values for age at which we want predictions, 
# and then call the generic predict() function for each model:

# Generate a sequence of age values spanning the range
import numpy as np    
age_grid = np.arange(df.age.min(), df.age.max()).reshape(-1,1) # [18,19,20,...,78,79]

# Generate test data
X_test = PolynomialFeatures(4).fit_transform(age_grid)

# Predict the value of the generated ages
pred1 = fit2.predict(X_test) # salary
pred2 = res.predict(X_test)  # Pr(wage>250)


# Finally, we plot the data and add the fit from the degree-4 polynomial.
# creating plots
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
fig.suptitle('Degree-4 Polynomial', fontsize=14)

# Scatter plot with polynomial regression line
ax1.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.3)
ax1.plot(age_grid, pred1, color = 'b')
ax1.set_ylim(ymin=0)

# Logistic regression showing Pr(wage>250) for the age range.
ax2.plot(age_grid, pred2, color='b')

# Rug plot showing the distribution of wage>250 in the training data.
# 'True' on the top, 'False' on the bottom.
ax2.scatter(df.age, y/5, s=30, c='grey', marker='|', alpha=0.7)

ax2.set_ylim(-0.01,0.21)
ax2.set_xlabel('age')
ax2.set_ylabel('Pr(wage>250|age)')


# Step functions

'''

In order to fit a step function, we use the cut() function:
    
retbins : bool, default False
    whether to return the bins or not. Useful when bins is provided as a scalar.
right : bool, default True
    Indicates whether bins includes the rightmost edge or not. If right == True (the default),
    then the bins [1, 2, 3, 4] indicate (1,2], (2,3], (3,4]. This argument is ignored when bins is an IntervalIndex.


'''
    
df_cut, bins = pd.cut(df.age, 4, retbins = True, right = True)
df_cut.value_counts(sort = False) # Returns object containing counts of unique values.

# Here cut() automatically picked the cutpoints at 33.5, 49, and 64.5 years of age. 
# We could also have specified our own cutpoints directly. Now let's create a set
# of dummy variables for use in the regression

df_steps = pd.concat([df.age, df_cut, df.wage], keys = ['age','age_cuts','wage'], axis = 1)

# Create dummy variables for the age groups
df_steps_dummies = pd.get_dummies(df_steps['age_cuts'])

# Statsmodels requires explicit adding of a constant (intercept)
df_steps_dummies = sm.add_constant(df_steps_dummies)

# Drop the (17.938, 33.5] category
df_steps_dummies = df_steps_dummies.drop(df_steps_dummies.columns[1], axis = 1)

df_steps_dummies.head(5)

'''

An now to fit the models! We dropped the age<33.5 category, so the intercept coefficient of $94,160 
can be interpreted as the average salary for those under 33.5 years of age. 
The other coefficients can be interpreted as the average additional salary for those in the other age groups.

'''

fit3 = sm.GLM(df_steps.wage, df_steps_dummies).fit()
fit3.summary().tables[1]

# Put the test data in the same bins as the training data.
bin_mapping = np.digitize(age_grid.ravel(), bins)

# Get dummies, drop first dummy category, add constant
X_test2 = sm.add_constant(pd.get_dummies(bin_mapping).drop(1, axis = 1))

# Predict the value of the generated ages using the linear model
pred2 = fit3.predict(X_test2)

# And the logistic model
clf2 = sm.GLM(y, df_steps_dummies,family=sm.families.Binomial(sm.families.links.logit))
res2 = clf2.fit()
pred3 = res2.predict(X_test2)

# Plot
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5))
fig.suptitle('Piecewise Constant', fontsize = 14)

# Scatter plot with polynomial regression line
ax1.scatter(df.age, df.wage, facecolor = 'None', edgecolor = 'k', alpha = 0.3)
ax1.plot(age_grid, pred2, c = 'b')

ax1.set_xlabel('age')
ax1.set_ylabel('wage')
ax1.set_ylim(ymin = 0)

# Logistic regression showing Pr(wage>250) for the age range.
ax2.plot(np.arange(df.age.min(), df.age.max()).reshape(-1,1), pred3, color = 'b')

# Rug plot showing the distribution of wage>250 in the training data.
# 'True' on the top, 'False' on the bottom.
ax2.scatter(df.age, y/5, s = 30, c = 'grey', marker = '|', alpha = 0.7)

ax2.set_ylim(-0.01, 0.21)
ax2.set_xlabel('age')
ax2.set_ylabel('Pr(wage>250|age)')


''' Splines '''
''' The main model to use here is patsy'''

data_x = df['age'] # Series (300,) and df[['age']] DataFrame (3000,1)
data_y = df['wage']

# Dividing data into train and validation datasets
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.33, random_state = 1)

from patsy import dmatrix

'''

patsy is a Python package for describing statistical models 
(especially linear models, or models that have a linear component) and building design matrice

patsy.dmatrix(formula_like, data={}, eval_env=0, NA_action='drop', return_type='matrix')
Construct a single design matrix given a formula_like and data.
'''

# Generating cubic spline with 3 knots at 25, 40 and 60
transformed_x = dmatrix("bs(train, knots=(25,40,60), degree=3, include_intercept=False)", {"train": train_x},return_type='dataframe')

# Fitting Generalised linear model on transformed dataset
fit1 = sm.GLM(train_y, transformed_x).fit()

# Generating cubic spline with 4 knots
transformed_x2 = dmatrix("bs(train, knots=(25,40,50,65),degree =3, include_intercept=False)", {"train": train_x}, return_type='dataframe')

# Fitting Generalised linear model on transformed dataset, using OLS 
# (since now we got the transformed dataset (since we ow have the basis function bi(x)))
fit2 = sm.GLM(train_y, transformed_x2).fit()

# Predictions on both splines
pred1 = fit1.predict(dmatrix("bs(valid, knots=(25,40,60), include_intercept=False)", {"valid": valid_x}, return_type='dataframe'))
pred2 = fit2.predict(dmatrix("bs(valid, knots=(25,40,50,65),degree =3, include_intercept=False)", {"valid": valid_x}, return_type='dataframe'))

# Calculating RMSE values
from sklearn.metrics import mean_squared_error
rms1 = np.sqrt(mean_squared_error(valid_y, pred1))
print(rms1) # -> 39.4

rms2 = np.sqrt(mean_squared_error(valid_y, pred2))
print(rms2) # -> 39.3


# We will plot the graph for 70 observations only
xp = np.linspace(valid_x.min(),valid_x.max(),70) # valid is the x-validation set

# Make some predictions
pred1 = fit1.predict(dmatrix("bs(xp, knots=(25,40,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))
 # Ci dessus: liste de substitution {"xp": xp}
pred2 = fit2.predict(dmatrix("bs(xp, knots=(25,40,50,65),degree =3, include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# Plot the splines and error bands
plt.figure()
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1) # alpha=0 transparent, alpha=1 opaque
plt.plot(xp, pred1, label='Specifying degree =3 with 3 knots')
plt.plot(xp, pred2, color='r', label='Specifying degree =3 with 4 knots')
plt.legend()
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
plt.show()

'''

We know that the behavior of polynomials that are fit to the data tends to be erratic near the boundaries. 
Such variability can be dangerous. These problems are resembled by splines, too. 
The polynomials fit beyond the boundary knots behave even more wildly than the corresponding global polynomials 
in that region. 
To smooth the polynomial beyond the boundary knots, we will use a special type of spline known as Natural Spline.

A natural cubic spline adds additional constraints, namely that the function is linear beyond the boundary knots. 
This constrains the cubic and quadratic parts there to 0, each reducing the degrees of freedom by 2. 
That’s 2 degrees of freedom at each of the two ends of the curve, reducing K+4 to K


'''

# Generating natural cubic spline
transformed_x3 = dmatrix("cr(train,df = 3)", {"train": train_x}, return_type='dataframe')

# We fit it
fit3 = sm.GLM(train_y, transformed_x3).fit()

# Prediction on validation set
pred3 = fit3.predict(dmatrix("cr(valid, df=3)", {"valid": valid_x}, return_type='dataframe'))

# Calculating RMSE value
rms = np.sqrt(mean_squared_error(valid_y, pred3))
print(rms) #-> 39.44

# We will plot the graph for 70 observations only
xp = np.linspace(valid_x.min(),valid_x.max(),70)
pred3 = fit3.predict(dmatrix("cr(xp, df=3)", {"xp": xp}, return_type='dataframe'))

# Plot the spline
plt.figure()
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(xp, pred3,color='g', label='Natural spline')
plt.legend()
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
plt.show()