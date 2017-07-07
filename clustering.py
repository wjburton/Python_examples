#clustering example in python using iris dataset and
# digit recognition

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import sklearn.cluster as clust
import re
from scipy.cluster.hierarchy import linkage, dendogram
from mpl_toolkits.mplot3d import Axes3D


iris = datasets.load_iris()
type(iris)
iris.head()

df = pd.DataFrame(iris.data)
df['target'] = iris.target

col_names = iris.feature_names
extract_name = lambda x: (re.search('(.*h).*', x) #find pattern
                          .group(1) #extract group
                          .replace(' ', '_')) #replace spaces with underscore

col_names = list(map(extract_name, col_names)) #map the function onto the list
col_names = dict(zip(df.columns, col_names)) #form a dictionary

df = df.rename(columns = col_names) # rename dataframe


def k_means_sse(k, X, seed)
    out = clust.KMeans(n_clusters=k, random_state=seed).fit(X)


X = df.iloc[:,0:3]
seed = 5

centroids = out.cluster_centers_
Axes3D.scatter(df.sepal_length, df.sepal_width, df.petal_length, zdir='z', s=20, c=None, depthshade=True)
#Digit recognition

iris_plot = plt.figure()
iris_plot = iris_plot.add_subplot(111,projection='3d')
iris_plot.scatter(df.sepal_length, df.sepal_width, df.petal_length, c = df.target)
iris_plot.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2],
           marker = "x", s=150, linewidths = 5, zorder = 100, c= ["g", "r", "b", "y"])
iris_plot.set_xlabel('Sepal Length')
iris_plot.set_ylabel('Sepal Width')
iris_plot.set_zlabel('Petal Length')