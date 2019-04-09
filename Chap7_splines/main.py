'''
https://patsy.readthedocs.io/en/latest/quickstart.html
https://patsy.readthedocs.io/en/latest/spline-regression.html
patsy is a Python package for describing statistical models 
(especially linear models, or models that have a linear component) 
and building design matrices.
'''



'''
For instance, if we have some variable y, 
and we want to regress it against 
some other variables x, a, b, and the interaction of a and b, 
then we simply write:
    
patsy.dmatrices("y ~ x + a + b + a:b", data)

>>>>>>>>>>>and Patsy takes care of building appropriate matrices.<<<<<<<<

BE CAREFUL: Patsy does not perform the regression, but construct a "nice" matrix
on which a regression can be performed easily

'''

from patsy import dmatrix, dmatrices
import numpy as np

# Let's generate random data from scratch:

# first 2 independent variables. These will be the 2 predictors:
n = 50 # number of observations

x1 = np.random.rand(n,1) # un peu inutile de generer des trucs aleatoires, mieux linspace
x2 = np.random.rand(n,1)

# and then the true response y = x1 +2*x2 for example
const = [1]*n # list [4,...,4]
const = (np.asarray(const)).reshape(-1,1) # array([4,...,4])

y = const+x1+2*x2

# Above is the true relationship between the predictors and the response
# Let's add some noise to y

mu, sigma = 0, 0.01 # mean and standard deviation
noise = np.random.normal(mu, sigma, n).reshape(-1,1) # generate n gaussian random numbers
y = y + noise

# Let's plot some graph
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(x1,y)
plt.xlabel('x1')
plt.ylabel('y = 1 + x1 + 2x2')
plt.show()

dmat = dmatrices("y ~ x1 + x2")
print(dmat)

'''

dmat contains 2 objects:
the 1st representing the left-hand side of our formula, ie y 
the 2nd representing the right-hand side.
Notice that an intercept term was automatically added to the right-hand side.
(To get rid of the intercept: dmatrix("x1 + x2 - 1", data))

Outputs are just ordinary numpy arrays with some extra metadata and a fancy __repr__ method attached, 
so we can pass them directly to a regression function like 
np.linalg.lstsq(a, b, rcond=-1)[source])
which solves the equation "b = ax" by computing a vector x that minimizes the Euclidean 2-norm || b - a x ||^2

(cf linear regression, what we want is find Beta in Y = X x Beta + Eps  such that ||Y - X x Beta ||^2  is minimal)

Other example operations:
    
dmatrix("x1 + np.log(x2 + 10)", data)
Out[13]: 
DesignMatrix with shape (8, 3)
Intercept        x1         np.log(x2 + 10)
    1           1.76405          2.29221
    1           0.40016          2.34282

'''

outcome, predictors = dmatrices("y ~ x1 + x2")
betas = np.linalg.lstsq(predictors, outcome)[0].ravel()

''' 

Splines

yi = b0(xi)+...+bj(xi)+...+bK(xi)

ex: spline cubique, we have:
intercept,x,x^2,x^3,h(x,khi1),h(x,khi2),h(x,khi3)
so 7 degrees of freedom (before fitting)

'''

plt.figure()
plt.title("B-spline basis example (degree=3)");

x = np.linspace(0., 1., 100)
y = dmatrix("bs(x, df=6, degree=3, include_intercept=True) -1", {"x": x}) 

'''

Voir la matrice precedente comme 
dmat = dmatrices("y ~ x1 + x2") (cf plus hait REGRESSION LINEAIRE) ou dmatrix("x1 + np.log(x2 + 10)", data)
meme idee, produit une matrice (n x nombre de predicteurs ) avec ce qu on veut dedans

We get rid of the [1,...1] column but we still have the "intercept", so now 6 degrees of freedom

'''

# Define some coefficients (these coeff won't be random when fit)
b = np.array([1.3, 0.6, 0.9, 0.4, 1.6, 0.7])

# Plot B-spline basis functions (colored curves) each multiplied by its coeff (the individual one)
exp = y*b # y: (100,6) b:(6,) exp: (100,6) the multiplication is ELEMENTWISE
plt.plot(x, exp);

# Plot the spline itself (sum of the basis functions, thick black curve)
plt.plot(x, np.dot(y, b), color='k', linewidth=3);

        
'''

Dummy example now: 
    
generate AND fit a cubic spline with the generated data


'''

N = 50
X = np.linspace(0,1,N).reshape(-1,1) # n points uniformly on [0,1]
X2 = X*X # elementwise multiplication
X3 = X*X*X
Y = 3+X+X2-2*X3 # the true response

mu, sigma = 0, 0.2 # mean and standard deviation
bruit = np.random.normal(mu, sigma, N).reshape(-1,1)

Y = Y+bruit # adding some noise

plt.figure()
plt.title("B-spline and Natural spline fitted to generated data");
plt.plot(X,Y,'k.')

# Cf Chap7, bsplines is actually transformed_x...
bsplines = dmatrix("bs(x, df=6, degree=2, include_intercept=True)-1", {"x": X}) # degree can be up to 5 but stay at cubic level

'''
degree=0: we fit with piecewise constant functions
degree=1: " piecewise linear "

'''
                  
                  
# Cf Chap 7, instead of using dmatrices and ~ to perform the regression,  we can also use sm.GLM
outcome, predictors = dmatrices("Y ~ bsplines -1") # remove the colums of 1 cause the splines produce already an intercept b0(xi) for each xi

# Cf Chap 7, alternative method...
betas = np.linalg.lstsq(predictors, outcome)[0].ravel()

# Plot
plt.plot(X, np.dot(bsplines, betas), color='r', linewidth=3, label='cubic spline');
plt.legend()
plt.xlabel('x')
plt.ylabel('y=3+x^2-2x^3+noise')

# Training MSE
MSE_training = ((Y-np.dot(bsplines, betas))**2).sum() # be careful, Y are the noisy observations
print(MSE_training) #3782

# R_squared
RSS = MSE_training # residual sum of squares, part de variance non expliuee (=epsilon aleatoire)
TSS = ((Y-np.mean(Y))**2).sum() #  total de la variance du modele (E[(X-E[X])^2)
R_squared = (TSS-RSS)/TSS  # proportion de variance expliquee par le modele
print(R_squared)
     
'''

Polynomial regression might actually give better result since the true response is generated from a polynomial function

'''
nsplines = dmatrix("cr(x, df=6) - 1", {"x": X})
outcome, predictors = dmatrices("Y ~ nsplines -1")
beta_ns = np.linalg.lstsq(predictors, outcome)[0].ravel()
plt.plot(X, np.dot(nsplines, beta_ns), color='b', linewidth=3, label='natural spline');
        
''' 

Natural splines seem to be slightly less wiggly

'''


'''

Now some tests on the intercept

'''

xx = np.linspace(0,1,n).reshape(-1,1) # (50,1)
yy = 2+3*xx
mu, sigma = 0, 0.5 # mean and standard deviation
noise = np.random.normal(mu, sigma, n).reshape(-1,1) # generate n gaussian random numbers
yy = yy + noise # (50,1)

plt.figure()
plt.plot(xx,yy,'b.')

'''
Below:
Construct a "nice" matrix with the intercept.
The matrix stored in pred
The out is just yy here

'''

out, pred = dmatrices("yy ~ xx") 

bb = np.linalg.lstsq(pred, out)[0].ravel().reshape(-1,1) # (2,1)
ee = np.dot(pred,bb)
plt.plot(xx,ee)

out_no_intercept, pred_no_intercept = dmatrices("yy ~ xx -1") 
bb_no_intercept = np.linalg.lstsq(pred_no_intercept, out_no_intercept)[0].ravel().reshape(-1,1) # (2,1)
ee_no_intercept = np.dot(pred_no_intercept,bb_no_intercept)
plt.plot(xx,ee_no_intercept)
