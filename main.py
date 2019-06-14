import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\I685383\\Desktop\\S\\Python tests\\DBCaseStudy\\project.csv', header=0, na_values='?')
df = df.dropna().reset_index(drop=True) # drop the possible observations with NA values and reindex the observations from 0

liste_df = list(df)


''' Summary statistics'''
pd.set_option('display.max_columns', None) # to inspect all the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # to suppress scientific notation
print(df.describe())
df.hist(figsize=(10,10))


''' Pairwise scatter plot to get side-by-side boxplots of the RAW data '''
from pandas.plotting import scatter_matrix
#scatter_matrix(df, alpha = 0.2, figsize = (10, 10), diagonal = 'hist')

''' Plot the position as a function of time '''
tab = df.values
tab = tab[:,1:]

for i in range(4):
    plt.figure()
    plt.plot(tab[:,i])
    plt.title(str(liste_df[i+1]))

# Let's isolate the std dev of the positions [already outputted  by the scatterplot]
# We can see position EEE is the wiggliest
wiggly_position = [np.std(tab[:,i]) for i in range(4)]


''' Create the Delta positions and the Returns''' 
n = tab.shape[0]
DeltaPosition = np.zeros((n-1,5))
for i in range(n-1):
    for j in range(0,4):
        DeltaPosition[i,j] =  tab[i+1,j] -tab[i,j]

Returns = np.zeros((n-1,5))
for i in range(n-1):
    for j in range(0,4):
        Returns[i,j] =  (tab[i+1,j+5] -tab[i,j+5])/tab[i,j+5]
        

''' Let's see if that better explained the data with a scatterplot ''' 
newData = pd.DataFrame(np.column_stack((DeltaPosition,Returns)))
scatter_matrix(newData.iloc[:1000,:], alpha = 0.2, figsize = (10, 10), diagonal = 'hist')
# Not really, but some linear relationships appear between the rates of Rate3 and Rate2 for example

''' Let's try to see if the [Delta position] is explained by the [Rate] or the [Return of the Rate]'''

import statsmodels.api as sm

X = Returns[:,0]
X = sm.add_constant(X)
Y = DeltaPosition[:,0]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=int(n/2), shuffle=True)

# Regression
ols = sm.OLS(Y_train, X_train)
ols_result = ols.fit()
print(ols_result.summary())
#Ypred = ols.predict(X_test)
#square_error = ((Y_test - Ypred)**2)
#print("MSE test = "+ str(np.mean(square_error))) # 26.74
                            
''' In the univariate case''' 

''' In the multivariate case''' 



''' Let us draw the P&l '''
earnings = np.zeros(n)

for i in range(n):
    earnings[i] = np.sum([tab[i,j]*tab[i,j+5] for j in range(4)])

plt.figure()
plt.plot(earnings)
plt.title("P&L")
