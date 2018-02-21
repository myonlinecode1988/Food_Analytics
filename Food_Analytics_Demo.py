"""Food Analytics for Epicurious.com
Data source: https://www.kaggle.com/hugodarwood/epirecipes

The idea of the project is explore the above data set and glean some insights"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
dataset = pd.read_csv('./epicurious-recipes-with-rating-and-nutrition/epi_r.csv')

# head
print(dataset.head(6))

#Remove NAs
dataset=dataset.dropna()

#Subsetting title','rating','calories','protein','fat' columns
recipes_with_macros=dataset[['title','rating','calories','protein','fat']];

#Adding carbohydate column. Calculate carbohydate in grams by using the following
#formula Carbs(in gm)=[Calories-9*fat(in gm)-4*protein(in gm)]/4 
#Note Caloriesin 1 gm of carb=4,Protein=4 & Fat=9
recipes_with_macros.at[:,'carbs']=recipes_with_macros.apply(lambda row: \
                   round((row['calories']-9*row['fat']-4*row['protein'])/4), axis=1)

#Remove negative carbs
recipes_with_macros=recipes_with_macros.loc[(recipes_with_macros['carbs'] > 0)]

#Adding percentagesof calories for protein,fat & carbs.
recipes_with_macros.at[:,'%protein']=recipes_with_macros.apply(lambda row: row['protein']*400/row['calories'], axis=1)
recipes_with_macros.at[:,'%fat']=recipes_with_macros.apply(lambda row: row['fat']*900/row['calories'], axis=1)
recipes_with_macros.at[:,'%carbs']=recipes_with_macros.apply(lambda row: row['carbs']*400/row['calories'], axis=1)


CARBS_RANGE=[10,30]
PROTEIN_RANGE=[40,50]
FAT_RANGE=[30,40]

healthy_recipes=recipes_with_macros[(recipes_with_macros['%carbs'] >CARBS_RANGE[0]) & \
                                    (recipes_with_macros['%carbs'] <CARBS_RANGE[1]) & \
                                    (recipes_with_macros['%protein'] >PROTEIN_RANGE[0]) & \
                                    (recipes_with_macros['%protein'] <PROTEIN_RANGE[1]) & \
                                    (recipes_with_macros['%fat'] >FAT_RANGE[0]) & \
                                    (recipes_with_macros['%fat'] <FAT_RANGE[1])]
#Display by lowest calories,highest rating
healthy_recipes=healthy_recipes.sort_values(['calories', 'rating'], ascending=[True, False])


X_KNN = dataset.iloc[:, 6:680].values
t=recipes[recipes['Cluster']==0]
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_KNN)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_KNN)

#Subsetting 'title'
recipes=dataset[['title']];
recipes.at[:,'Cluster']=y_kmeans

print recipes[recipes['Cluster']==1]

X = dataset.iloc[:, 2:680].values
y = dataset.iloc[:, 1].values

# Encode th output
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

plt.figure()
plt.hist(y_test , bins=np.arange(9)-0.5)
plt.show()

###########Try out a few algorithms###########################################
##############################################################################
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = classifier_KNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_true=y_test, y_pred=y_pred_KNN)

print "Accuracy KNN",cm_KNN.trace()/float(cm_KNN.sum())


################################################################################
#Create a fake/naive  confusion matrix################################################
p1=(dataset.groupby(encoded_y).size()).values
p1= list(p1/float(p1.sum()))
y_sample=[]
for sample in xrange(0,30):
    y_fake=[]
    for i,j in enumerate(y_test):
        y_fake.append(np.random.choice(np.arange(0, 8), p=p1))
    
    cm_FAKE = confusion_matrix(y_test, y_fake)
    y_sample.append(cm_FAKE.trace()/float(cm_FAKE.sum()))

print "Average Accuracy fake model",np.asarray(y_sample).mean(), np.asarray(y_sample).std()

################################################################################
## Let's try some dimension reduction techniques
########################Dimesnion Reduction###################################

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

plt.figure()
plt.bar(range(0,len(explained_variance)),explained_variance)
plt.show()

##############################################################################
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RFC = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0,n_jobs=-1)
classifier_RFC.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RFC = classifier_RFC.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RFC = confusion_matrix(y_test, y_pred_RFC)
print ("Accuracy RFC=",cm_RFC.trace()/cm_RFC.sum())
##############################################################################
y_accuracy=[]
# Applying PCA
for i in range(1,678,10):
    pca = PCA(n_components = i)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.transform(X_test)
    
    classifier_RFC = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0,n_jobs=-1)
    classifier_RFC.fit(X_train1, y_train)
    
    # Predicting the Test set results
    y_pred_RFC = classifier_RFC.predict(X_test1)
    
    # Making the Confusion Matrix
    cm_RFC = confusion_matrix(y_test, y_pred_RFC)
    y_accuracy.append(cm_RFC.trace()/float(cm_RFC.sum()))

plt.figure()
plt.plot(range(1,678,10),y_accuracy)
plt.show()
##############################################################################
pca = PCA(n_components = 100)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)

classifier_RFC = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, n_jobs =-1)
classifier_RFC.fit(X_train1, y_train)

# Predicting the Test set results
y_pred_RFC = classifier_RFC.predict(X_test1)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_RFC, X = X_train1, y = y_train, cv = 10)
print ("Average Accuracy Random Forest+PCA=",accuracies.mean(),"std=", accuracies.std())

from sklearn.model_selection import GridSearchCV

# use a full grid over all parameters
param_grid = {"max_features": range(100,200,10),           
              "criterion": ["entropy","gini"]}

# run grid search
grid_search = GridSearchCV(classifier_RFC, param_grid=param_grid,cv=10,n_jobs=-1,scoring = 'accuracy')
grid_search.fit(X_train, y_train)



