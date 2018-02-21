# Using K-Nearest Neighbors (K-NN) and RandomForest

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./epicurious-recipes-with-rating-and-nutrition/epi_r.csv')
dataset=dataset.dropna()



X = dataset.iloc[:, 2:680].values
y = dataset.iloc[:, 1].values

# Encode th output
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

#Summarize thre dataset
# shape
print(dataset.shape)

# head
print(dataset.head(6))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby(encoded_y).size())


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


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
cm_KNN = confusion_matrix(y_test, y_pred_KNN)

##############################################################################
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RFC.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RFC = classifier_RFC.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RFC = confusion_matrix(y_test, y_pred_RFC)
##############################################################################
#Model selection:Grid Search KNN##############################################
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors': [5, 7, 9]}
             ]
grid_search = GridSearchCV(estimator = classifier_KNN,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print "Accuracy KNN",cm_KNN.trace()/float(cm_KNN.sum())
print "Accuracy RFC",cm_RFC.trace()/float(cm_RFC.sum())
########################Dimesnion Reduction###################################

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

plt.bar(range(0,len(explained_variance)),explained_variance)
plt.show()
##############################################################################
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = classifier_KNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
print "Accuracy KNN",cm_KNN.trace()/float(cm_KNN.sum())



##############################################################################
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RFC.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RFC = classifier_RFC.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RFC = confusion_matrix(y_test, y_pred_RFC)
print "Accuracy RFC",cm_RFC.trace()/float(cm_RFC.sum())
plt.bar(range(0,len(classifier_RFC.feature_importances_)),classifier_RFC.feature_importances_)
plt.show()
################################################################################
#Create a dumb confusion matrix################################################
p1=(dataset.groupby(encoded_y).size()).values
p1= list(p1/float(p1.sum()))
y_sample=[]
for sample in xrange(0,30):
    y_fake=[]
    for i,j in enumerate(y_test):
        y_fake.append(np.random.choice(np.arange(0, 8), p=p1))
    
    cm_FAKE = confusion_matrix(y_test, y_fake)
    y_sample.append(cm_FAKE.trace()/float(cm_FAKE.sum()))

print "Avergea Accuracy fake model",np.asarray(y_sample).mean(), np.asarray(y_sample).std()

################################################################################
#Create a dumb confusion matrix################################################

y_accuracy=[]
# Applying PCA
for i in range(1,678,10):
    pca = PCA(n_components = i)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.transform(X_test)
    
    classifier_RFC = RandomForestClassifier(n_estimators = 130, criterion = 'entropy', random_state = 0)
    classifier_RFC.fit(X_train1, y_train)
    
    # Predicting the Test set results
    y_pred_RFC = classifier_RFC.predict(X_test1)
    
    # Making the Confusion Matrix
    cm_RFC = confusion_matrix(y_test, y_pred_RFC)
    y_accuracy.append(cm_RFC.trace()/float(cm_RFC.sum()))

plt.plot(range(1,678,10),y_accuracy)

range(1,678,10)

################################################################################
# Fitting OVRC to the Training set
from sklearn.multiclass import OneVsRestClassifier
classifier_OVRC_KNN = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),n_jobs=-1)
classifier_OVRC_KNN.fit(X_train, y_train)

# Predicting the Test set results
y_pred_OVRC_KNN = classifier_OVRC_KNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_OVRC_KNN = confusion_matrix(y_test, y_pred_OVRC_KNN)
print "Accuracy OVRC",cm_OVRC_KNN.trace()/float(cm_OVRC_KNN.sum())

################################################################################
# Fitting OVRC-Random Forest to the Training set
from sklearn.multiclass import OneVsRestClassifier
classifier_OVRC = OneVsRestClassifier(RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),n_jobs=-1)
classifier_OVRC.fit(X_train, y_train)

# Predicting the Test set results
y_pred_OVRC = classifier_OVRC.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_OVRC = confusion_matrix(y_test, y_pred_OVRC)
print "Accuracy OVRC",cm_OVRC.trace()/float(cm_OVRC.sum())
#y_score = classifier_OVRC.fit(X_train, y_train).decision_function(X_test)

#############################MULTI_CLASS ROC curve#############################
#ROC curve for class 7
#y_new_pred_OVRC=y_pred_OVRC
#y_new_test=y_test
#for idx, item in enumerate(y_new_pred_OVRC):
#    if (item!=6):
#        y_new_pred_OVRC[idx]=0
#        
#for idx, item in enumerate(y_new_test):
#    #print item
#    if (item!=6):
#        y_new_test[idx]=0
#
#from sklearn.metrics import roc_curve, auc            
#fpr, tpr, thresholds = roc_curve(y_new_test, y_new_pred_OVRC,pos_label=6)
#roc_auc = auc(np.sort(y_new_test), np.sort(y_new_pred_OVRC))
## Plot ROC curve
#plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
#plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.xlabel('False Positive Rate or (1 - Specifity)')
#plt.ylabel('True Positive Rate or (Sensitivity)')
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc="lower right")

from sklearn.preprocessing import label_binarize
# Binarize the output
y = label_binarize(encoded_y, classes=[0, 1, 2, 3, 4, 5, 6, 7])
n_classes = y.shape[1]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting OVRC-Random Forest to the Training set
from sklearn.multiclass import OneVsRestClassifier
classifier_OVRC = OneVsRestClassifier(RandomForestClassifier(n_estimators = 130, criterion = 'gini', random_state = 0),n_jobs=-1)
classifier_OVRC.fit(X_train, y_train)

# Predicting the Test set results
y_pred_OVRC = classifier_OVRC.predict(X_test)


################################################################################
################################################################################
################################################################################
################################################################################



from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

classifier_RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RFC.fit(X_train, y_train)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 200),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

## run randomized search
#n_iter_search = 2
#random_search = RandomizedSearchCV(classifier_RFC, param_distributions=param_dist,
#                                   n_iter=n_iter_search,cv=10,n_jobs=-1)
#
#start = time()
#random_search.fit(X_train, y_train)
#print("RandomizedSearchCV took %.2f seconds for %d candidates"
#      " parameter settings." % ((time() - start), n_iter_search))
#report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_features": range(100,200,10),           
              "criterion": ["gini"]}

# run grid search
grid_search = GridSearchCV(classifier_RFC, param_grid=param_grid,cv=10,n_jobs=-1,scoring = 'accuracy')
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_



################################################################################
################################################################################
################################################################################
################################################################################






from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_OVRC[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_OVRC.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])




plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

from scipy.stats import randint as sp_randint

