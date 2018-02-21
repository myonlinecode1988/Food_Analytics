#Machine Learning Demo for Insight Data Science

#%reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


dataset = pd.read_csv('./epicurious-recipes-with-rating-and-nutrition/epi_r.csv')
dataset=dataset.dropna()

#Getting a feel of Data
#print dataset.head(6)
#print "Total RowsxColumsn",dataset.shape


#Distribution of the ratings
#What is the composition of vegetarian and non-vegetarian recipes ?
## What is the relation between rating and calorie of the recipes?
#Will there be any correlation between Nutrition features and Rating's of Recipes?
#What is the composition of vegetarian and non-vegetarian recipes in 5 start rating?
#What are common fruits used in 5 stars rated recipes?
#
##https://www.kaggle.com/veereshelango/curious-on-epicurious-recipes
#https://www.kaggle.com/amunnelly/nutrition-in-dairy-v-dairy-free-recipes
#

##Get index location by name
#rownames=list(dataset.axes[1]);
#rownames.index("turkey")




#X = dataset.iloc[:, 6:680].values

## Using the elbow method to find the optimal number of clusters
#from sklearn.cluster import KMeans
#wcss = []
#for i in range(1, 11):
#    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)
#plt.plot(range(1, 11), wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()
#
## Fitting K-Means to the dataset
#kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
#y_kmeans = kmeans.fit_predict(X)
#
#
#X_train=X;
## Applying PCA
#from sklearn.decomposition import PCA
#pca = PCA()
#X_train = pca.fit_transform(X_train)
##X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_
#
##def raise_window(figname=None):
##    if figname: plt.figure(figname)
##    cfm = plt.get_current_fig_manager()
##    cfm.window.activateWindow()
##    cfm.window.raise_()  
#
#plt.bar(range(0,len(explained_variance)),explained_variance)
#plt.show()
##raise_window()
#
### Visualising the Training set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_kmeans
#
###Random subsample
##index=np.random.randint(15000,size=500)
##X_set=X_set[index];
##y_set=y_set[index];
#
#
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
##plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
##             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
#
#for i, j in enumerate(np.unique(y_set)):
#    print i,j
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green','blue'))(i), label = j)
#plt.title('PCA(colored by k-means)')
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#plt.legend()
#plt.show()
#
####


X = dataset.iloc[:, 2:680].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Fitting Multiple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = regressor.predict(X_test)
#
## Predicting the train set results
#y_train_model = regressor.predict(X_train)
#
from sklearn.metrics import mean_squared_error
from math import sqrt
#
#rms_test1 = sqrt(mean_squared_error(y_test, y_pred))
#rms_test2 = sqrt(mean_squared_error(y_train, y_train_model))
#
###Pooor Fit
#
#import statsmodels.formula.api as sm
#X_train_ones = np.append(arr = np.ones((12691, 1)).astype(int), values = X_train, axis = 1)
#X_test_ones = np.append(arr = np.ones((3173, 1)).astype(int), values = X_test, axis = 1)
#
#def backwardElimination(x,X_test,sl):
#    numVars = len(x[0])
#    for i in range(0, numVars):
#        #print "Shape",X.shape,len(y)
#        regressor_OLS = sm.OLS(y_train, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        if maxVar > sl:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    x = np.delete(x, j, 1)
#                    X_test = np.delete(X_test, j, 1)
#    regressor_OLS.summary()
#    return (x,X_test)
# 
#    
#SL = 0.10
#X_opt = X_train_ones;
#X_test_opt=X_test_ones;
#(X_Modeled,X_Modeled_test) = backwardElimination(X_opt, X_test_ones,SL)
#
#regressor1 = LinearRegression()
#regressor1.fit(X_Modeled, y_train)
#
## Predicting the Test set results
#y_pred = regressor1.predict(X_Modeled_test)
#
## Predicting the train set results
#y_train_model = regressor1.predict(X_Modeled)
#
#rms_test1 = sqrt(mean_squared_error(y_test, y_pred))
#rms_test2 = sqrt(mean_squared_error(y_train, y_train_model))
#
## Fitting Random Forest Regression to the dataset
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
#regressor.fit(X_train, y_train)
#
#y_pred_train = regressor.predict(X_train)
#y_pred_test = regressor.predict(X_test)
#
#rms_train = sqrt(mean_squared_error(y_train, y_pred_train))
#rms_test = sqrt(mean_squared_error(y_test, y_pred_test))

from sklearn.preprocessing import LabelEncoder
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

