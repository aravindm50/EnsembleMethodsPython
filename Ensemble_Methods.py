# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:01:27 2019

@author: Aravind
"""

## Bagging with Decision Tree and KNN

import itertools
import numpy as np

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves, plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import load_breast_cancer

# Clean Data
cancer = load_breast_cancer()
X,y = cancer.data[:,0:2],cancer.target


#Weak Classifiers
clf1 = DecisionTreeClassifier(criterion="gini",max_depth = 1)
clf2 = KNeighborsClassifier(n_neighbors = 1,metric = "euclidean")
clf3 = RandomForestClassifier(random_state=42)

bagging1 = BaggingClassifier(base_estimator = clf1, n_estimators =10,
                             max_samples = 0.8, max_features = 0.8)
bagging2 = BaggingClassifier(base_estimator = clf2, n_estimators = 10,
                             max_samples = 0.8, max_features = 0.8 )
adaboost = AdaBoostClassifier(base_estimator= clf1, n_estimators= 10)

# Classifier List
label = ['Decision Tree','K-NN','Random Forest','Bagging Tree','Bagging K-NN',
         'Adaboost']
clf_list = [clf1, clf2,clf3, bagging1, bagging2,adaboost]

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,3)
grid = itertools.product([0,1,2],repeat = 2)

for clf, label, grd in zip(clf_list, label, grid):
    scores = cross_val_score(clf, X,y, cv = 3,scoring = "accuracy")
    print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    
    # Fit with weak classifers
    clf.fit(X,y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf = clf, legend = 2)
    plt.title(label)
    
plt.show()

# Train and test split for boosting and bagging classifiers
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

plt.figure()
plot_learning_curves(X_train, y_train, X_test, y_test, bagging1, print_model = False, style = 'ggplot')
plt.show()

# using numbers between  1 to 100 for adaboost and bagging
num_est = map(int,np.linspace(1,100,20))
bg_clf_cv_mean = []
bg_clf_cv_std = []
ada_clf_mean= []
ada_clf_std = []
for n_est in num_est :
    bg_clf = BaggingClassifier(base_estimator = clf1, n_estimators = n_est, 
                               max_samples=0.8, max_features= 0.8)
    adaboost_clf = AdaBoostClassifier(base_estimator=clf1, n_estimators=n_est)
    scores = cross_val_score(bg_clf, X, y, cv = 3, scoring = "accuracy")
    scores_ada = cross_val_score(adaboost_clf, X,y,cv = 3, scoring = "accuracy")
    ada_clf_mean.append(scores_ada.mean())
    ada_clf_std.append(scores_ada.std())
    bg_clf_cv_mean.append(scores.mean())
    bg_clf_cv_std.append(scores.std())
    
plt.figure()
(_, caps, _) = plt.errorbar(np.linspace(1,100,20), bg_clf_cv_mean,yerr = bg_clf_cv_std, c="blue", fmt = "-o", capsize = 5 )
for cap in caps:
    cap.set_markeredgewidth(1)
plt.ylabel('Accuracy')
plt.xlabel("Ensemble size")
plt.title("Bagging Cancer")
plt.show()

plt.figure()
(_, caps, _) = plt.errorbar(np.linspace(1,100,20), ada_clf_mean,yerr = ada_clf_std, c="red", fmt = "-o", capsize = 5 )
for cap in caps:
    cap.set_markeredgewidth(1)
plt.ylabel('Accuracy')
plt.xlabel("Ensemble size")
plt.title("Adaboost Cancer")
plt.show()


