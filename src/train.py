
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#import sklearn
from sklearn.preprocessing import StandardScaler

#import self-defined utilities
from utils import get_best_classifier, get_train_test_set

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
import parameters as pm
from oof_stack import StackingClassifier
from random import randint


# # Stacking

# # Parameters Tuning
# 
# 1. Logistic Regression
# 2. Kernel SVC
# 3. Random Forest Classifier
# 4. Extra Trees Classifier
# 5. AdaBoost Tree
# 6. XGB Classifier
# 7. K-Nearest Neighbors

# ## Two-Level Stack

second_level_stack = Stacking_Classifier()
file_dirs = ['basics/',
             'band/',
             'final/']

model_params = [('LR', LogisticRegression(), pm.logistic_param),
                ('SVC', SVC(), pm.svc_param),
                ('DT', DecisionTreeClassifier(), pm.dt_param),
                ('RF', RandomForestClassifier(), pm.rf_param),
                ('ET', ExtraTreesClassifier(), pm.extra_param),
                ('Ada', AdaBoostClassifier(), pm.ada_param),
                ('XGBT', XGBClassifier(), pm.xgb_param),
                ('KNN', KNeighborsClassifier(), pm.knn_param)]

S_test2 = np.zeros(418)
for file_dir in file_dirs:
    first_level_stack = StackingClassifier()
    X_train, y_train, X_test = get_train_test_set('../data/preprocessed/' + file_dir,
                                                  'Survived')

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    for name, model, param in model_params:
        base = get_best_classifier(X_train, y_train, model, param, n_iter=500)
        first_level_stack.add_predictionCV(X_train, y_train, X_test, base, name)


    S_train, S_test = first_level_stack.get_train_test_set()
    print(pd.DataFrame(S_train).corr())
    
    stacker = get_best_classifier(S_train, y_train, XGBClassifier(), pm.xgb_param, n_iter=2000, n_folds=10)
    S_test2 = S_test2 + stacker.predict(S_test)

y_pred = (S_test2 > 1.5).astype(int)

# In[4]:

test_df = pd.read_csv('../data/input/test.csv')
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                           'Survived': y_pred})
submission.to_csv('../output/submission30.csv', index=False)



