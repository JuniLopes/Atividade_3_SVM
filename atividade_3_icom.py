# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:54:19 2021

# Equipe:
# *   Aline
# *   Irlailton
# *   Juliane
# *   Rubens Lopes

"""

import numpy as np
parkinson = parkison = np.genfromtxt('D:/OneDrive/Faculdade/S7/parkinson/parkinson_formated.csv', delimiter = ',')
type(parkison)

features = parkison[:,:754]
features

targets = parkison[:, 754]
targets

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_norm = min_max_scaler.fit_transform(features)

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import make_scorer

param_grid = {'kernel': ['rbf','linear', 'poly']}
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score),'Precision': make_scorer(precision_score),'Recall': make_scorer(recall_score),'F1': make_scorer(f1_score)}

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svm = SVC()
gs = GridSearchCV(svm, cv=10,param_grid=param_grid, refit='AUC', return_train_score=True, scoring=scoring)
gs.fit(features_norm, targets)
results = gs.cv_results_

acc = results['mean_test_Accuracy'][1]
f1 = results['mean_test_F1'][1]
prec = results['mean_test_Precision'][1]
rec = results['mean_test_Recall'][1]
