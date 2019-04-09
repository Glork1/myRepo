import numpy as np
import pandas as pd

''' PCA '''


# Loading data
df = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap10\\USArrests.csv",header=0, na_values='NA').drop('Unnamed: 0', axis = 1)
df = df.dropna().reset_index(drop=True)

'''

(50,4)

50 states with 4 features 
Murder,Assault,Rape: numeric per 100 000
UrbanPop: percentage of the population living in urban areas

'''


# Summary statistics
pd.set_option('display.max_columns', None) # to inspect all the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # to suppress scientific notation
print(df.describe()) 

'''

Generate a summary for each column. 25% etc return the 25% percentile. df.describe('percentiles=None' to remove it)

       Murder  Assault  UrbanPop   Rape
count  50.000   50.000    50.000 50.000
mean    7.788  170.760    65.540 21.232
std     4.356   83.338    14.475  9.366
min     0.800   45.000    32.000  7.300
25%     4.075  109.000    54.500 15.075
50%     7.250  159.000    66.000 20.100
75%    11.250  249.000    77.750 26.175
max    17.400  337.000    91.000 46.000

Observations: we can see that the means are quite different, as well as the variance

Assault has by bar the largest mean and the largest variance.
So if we don't scale the variable, the most of the principal components that we observe would be driven by the Assault variable.
Important to standardize variables before doing the PCA
'''

df.hist(figsize=(10,10)) # plot histogram for each of the varuable

# Pairwise scatter plot to get side-by-side boxplots
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha = 0.2, figsize = (10, 10), diagonal = 'hist')

# Standardizin following the comments above
from sklearn.preprocessing import scale
X = pd.DataFrame(scale(df), index=df.index, columns=df.columns)

# The loading vectors
from sklearn.decomposition import PCA
pca_loadings = pd.DataFrame(PCA().fit(X).components_.T, index=df.columns, columns=['V1', 'V2', 'V3', 'V4'])

print(pca_loadings)

'''

            V1     V2     V3     V4
Murder   0.536  0.418 -0.341  0.649
Assault  0.583  0.188 -0.268 -0.743
UrbanPop 0.278 -0.873 -0.378  0.134
Rape     0.543 -0.167  0.818  0.089

'''

# Fit the PCA model and transform X to get the principal components
pca = PCA()
df_plot = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.index)

import matplotlib.pyplot as plt

plt.figure()
fig , ax1 = plt.subplots(figsize=(9,7))

ax1.set_xlim(-3.5,3.5)
ax1.set_ylim(-3.5,3.5)

# Plot Principal Components 1 and 2
for i in df_plot.index:
    ax1.annotate(i, (df_plot.PC1.loc[i], -df_plot.PC2.loc[i]), ha='center')

# Plot reference lines
ax1.hlines(0,-3.5,3.5, linestyles='dotted', colors='grey')
ax1.vlines(0,-3.5,3.5, linestyles='dotted', colors='grey')

ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
    
# Plot Principal Component loading vectors, using a second y-axis.
ax2 = ax1.twinx().twiny() 

ax2.set_ylim(-1,1)
ax2.set_xlim(-1,1)
ax2.tick_params(axis='y', colors='orange')
ax2.set_xlabel('Principal Component loading vectors', color='orange')

# Plot labels for vectors. Variable 'a' is a small offset parameter to separate arrow tip and text.
a = 1.07  
for i in pca_loadings[['V1', 'V2']].index:
    ax2.annotate(i, (pca_loadings.V1.loc[i]*a, -pca_loadings.V2.loc[i]*a), color='orange')

# Plot vectors
ax2.arrow(0,0,pca_loadings.V1[0], -pca_loadings.V2[0])
ax2.arrow(0,0,pca_loadings.V1[1], -pca_loadings.V2[1])
ax2.arrow(0,0,pca_loadings.V1[2], -pca_loadings.V2[2])
ax2.arrow(0,0,pca_loadings.V1[3], -pca_loadings.V2[3])


# Standard deviation of the four principal components

print(np.sqrt(pca.explained_variance_)) #[1.5908673  1.00496987 0.6031915  0.4206774 ]
print(pca.explained_variance_) #[2.53085875 1.00996444 0.36383998 0.17696948]
print(pca.explained_variance_ratio_) #[0.62006039 0.24744129 0.0891408  0.04335752]

# Choosing the right number of components
plt.figure(figsize=(7,5))

plt.plot([1,2,3,4], pca.explained_variance_ratio_, '-o', label='Individual component')
plt.plot([1,2,3,4], np.cumsum(pca.explained_variance_ratio_), '-s', label='Cumulative')

plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4])
plt.legend(loc=2);

''' Clustering '''

''' K means clustering '''

# Generate data
np.random.seed(2)

#X = np.random.normal(0,1,100).reshape(-1,2) # <- effectivily worse 
X = np.random.standard_normal((50,2))

X[:25,0] = X[:25,0] + 3
X[:25,1] = X[:25,1] - 3

plt.figure()
plt.scatter(X[:,0],X[:,1])

# K = 2
from sklearn.cluster import KMeans
km1 = KMeans(n_clusters=2, n_init=20)
km1.fit(X)

'''
class sklearn.cluster.KMeans(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001 ...

n_clusters : int, optional, default: 8

    The number of clusters to form as well as the number of centroids to generate.

init : {‘k-means++’, ‘random’ or an ndarray}

    Method for initialization, defaults to ‘k-means++’:

    ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.

    ‘random’: choose k observations (rows) at random from data for the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

n_init : int, default: 10

    Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.


'''

'''
clustering not perfect cf labels_km1
'''

np.random.seed(4)
km2 = KMeans(n_clusters=3, n_init=20)
km2.fit(X)

print(pd.Series(km2.labels_).value_counts())

'''
0    21
1    19
2    10

'''
print(km2.cluster_centers_)

'''
[[ 2.82805911 -3.11351797]
 [-0.34608792  0.5592591 ]
 [ 0.72954539 -1.57251836]]
'''

# Sum of distances of samples to their closest cluster center.
print(km2.inertia_) #66.91823062225033

# Some plots
plt.figure()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

ax1.scatter(X[:,0], X[:,1], s=20, c=km1.labels_, cmap=plt.cm.prism) 
ax1.set_title('K-Means Clustering Results with K=2')
ax1.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2)

ax2.scatter(X[:,0], X[:,1], s=40, c=km2.labels_, cmap=plt.cm.prism) 
ax2.set_title('K-Means Clustering Results with K=3')
ax2.scatter(km2.cluster_centers_[:,0], km2.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);

''' K means clustering '''
from scipy.cluster import hierarchy

plt.figure()

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,18))

for linkage, cluster, ax in zip([hierarchy.complete(X), hierarchy.average(X), hierarchy.single(X)], ['c1','c2','c3'],[ax1,ax2,ax3]):
    cluster = hierarchy.dendrogram(linkage, ax=ax, color_threshold=0)

ax1.set_title('Complete Linkage')
ax2.set_title('Average Linkage')
ax3.set_title('Single Linkage');


''' Lab Genes NCI60 '''

'''
Each cell line is labeled with a cancer type. We do not make use of the
cancer types in performing PCA and clustering, as these are unsupervised
techniques. But after performing PCA and clustering, we will check to
see the extent to which these cancer types agree with the results of these
unsupervised techniques.

'''

# Loading data
dg = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap10\\NCI60_X.csv",header=0, na_values='NA').drop('Unnamed: 0', axis = 1)
dg.columns = np.arange(dg.columns.size)
dg.info()

X = pd.DataFrame(scale(dg))
print(X.shape) # (64,6830)

Y = pd.read_csv("C:\\Users\\I685383\\Desktop\\S\\Python tests\\Chap10\\NCI60_y.csv", usecols=[1], skiprows=1, names=['type']) # (64, 1)
# skiprows = 1 : Line numbers to skip (0-indexed) 
print(Y.shape)
print(Y.type.value_counts())

'''
RENAL          9
NSCLC          9
MELANOMA       8
BREAST         7
COLON          7
OVARIAN        6
LEUKEMIA       6
CNS            5
PROSTATE       2
MCF7D-repro    1
K562B-repro    1
K562A-repro    1
MCF7A-repro    1
UNKNOWN        1
Name: type, dtype: int64
'''
pca2 = PCA()
dg_plot = pd.DataFrame(pca2.fit_transform(X)) # Fit the model with X and apply the dimensionality reduction on X

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))

color_idx = pd.factorize(Y.type)[0]

'''
pd.fatcorize: encode the object as an enumerated type or categorical variable.

Y
	type
0	CNS
1	CNS
2	CNS

Return array([0, 0, 1, 2, 0]) for exemple

The observations (cell lines) corresponding to a given
cancer type will be plotted in the same color, so that we can see to what
extent the observations within a cancer type are similar to each other

So it assigns a color to each of the 64 cell lines, based on the cancer
type to which it corresponds.
'''
cmap = plt.cm.hsv

# Left plot
ax1.scatter(dg_plot.iloc[:,0], -dg_plot.iloc[:,1], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax1.set_ylabel('Principal Component 2')

# Right plot
ax2.scatter(dg_plot.iloc[:,0], dg_plot.iloc[:,2], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax2.set_ylabel('Principal Component 3')

'''
We see that:
on the whole, cell lines
corresponding to a single cancer type do tend to have similar values on the
first few principal component score vectors. This indicates that cell lines
from the same cancer type tend to have pretty similar gene expression
levels.
'''

##### Legends of the 2 graphs
# Custom legend for the classes (y) since we do not create scatter plots per class (which could have their own labels).
handles = []
labels = pd.factorize(Y.type.unique())

import matplotlib as mpl

norm = mpl.colors.Normalize(vmin=0.0, vmax=14.0)

for i, v in zip(labels[0], labels[1]):
    handles.append(mpl.patches.Patch(color=cmap(norm(i)), label=v, alpha=0.5))

ax2.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# xlabel for both plots
for ax in fig.axes:
    ax.set_xlabel('Principal Component 1')
    
pd.DataFrame([dg_plot.iloc[:,:5].std(axis=0, ddof=0).as_matrix(),
              pca2.explained_variance_ratio_[:5],
              np.cumsum(pca2.explained_variance_ratio_[:5])],
             index=['Standard Deviation', 'Proportion of Variance', 'Cumulative Proportion'],
             columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

#####

plt.figure()

dg_plot.iloc[:,:10].var(axis=0, ddof=0).plot(kind='bar', rot=0)
plt.ylabel('Variances');
          
fig , (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

# Left plot
ax1.plot(pca2.explained_variance_ratio_, '-o')
ax1.set_ylabel('Proportion of Variance Explained')
ax1.set_ylim(ymin=-0.01)

# Right plot
ax2.plot(np.cumsum(pca2.explained_variance_ratio_), '-ro')
ax2.set_ylabel('Cumulative Proportion of Variance Explained')
ax2.set_ylim(ymax=1.05)

for ax in fig.axes:
    ax.set_xlabel('Principal Component')
    ax.set_xlim(-1,65)
    
''' Clustering 
We now proceed to hierarchically cluster the cell lines in the NCI60 data,
with the goal of finding out whether or not the observations cluster into
distinct types of cancer. To begin, we standardize the variables to have
mean zero and standard deviation one. As mentioned earlier, this step is
optional and should be performed only if we want each gene to be on the
same scale.'''

X= pd.DataFrame(scale(dg), index=Y.type, columns=dg.columns)

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,20))

for linkage, cluster, ax in zip([hierarchy.complete(X), hierarchy.average(X), hierarchy.single(X)],
                                ['c1','c2','c3'],
                                [ax1,ax2,ax3]):
    cluster = hierarchy.dendrogram(linkage, labels=X.index, orientation='right', color_threshold=0, leaf_font_size=10, ax=ax)

ax1.set_title('Complete Linkage')
ax2.set_title('Average Linkage')
ax3.set_title('Single Linkage');

# We choose now a complete linkage             
plt.figure(figsize=(10,20))
cut4 = hierarchy.dendrogram(hierarchy.complete(X),labels=X.index, orientation='right', color_threshold=140, leaf_font_size=10)
plt.vlines(140,0,plt.gca().yaxis.get_data_interval()[1], colors='r', linestyles='dashed');
          
'''
There are some clear patterns. All the leukemia cell lines fall in cluster 3,
while the breast cancer cell lines are spread out over three different clusters.
'''
          
          
''' K-means '''
np.random.seed(2)
km4 = KMeans(n_clusters=4, n_init=50)
km4.fit(X)

# Observations per KMeans cluster
print(pd.Series(km4.labels_).value_counts().sort_index())

''' Hierarchical '''
# Observations per Hierarchical cluster
cut4b = hierarchy.dendrogram(hierarchy.complete(X), truncate_mode='lastp', p=4, show_leaf_counts=True)

'''
Rather than performing hierarchical clustering on the entire data matrix,
we can simply perform hierarchical clustering on the first few principal
component score vectors, as follows:
'''
# Hierarchy based on Principal Components 1 to 5
plt.figure(figsize=(10,20))
pca_cluster = hierarchy.dendrogram(hierarchy.complete(dg_plot.iloc[:,:5]), labels=Y.type.values, orientation='right', color_threshold=100, leaf_font_size=10)

cut4c = hierarchy.dendrogram(hierarchy.complete(dg_plot), truncate_mode='lastp', p=4,show_leaf_counts=True)
# See also color coding in plot above.