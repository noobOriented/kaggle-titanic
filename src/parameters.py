import numpy as np

seed = 1121

logistic_param = {'penalty': ('l1', 'l2'),
                  'dual': [False],
                  'C': np.logspace(-8, 4, 13, base=2)}

svc_param = {'kernel': ('rbf', 'poly', 'sigmoid'),
             'C': np.logspace(-4, 4, 9, base=2),
             'gamma': np.logspace(-6, 0, 7, base=2),
             'random_state': [seed]}

svc_prob_param = {'kernel': ('rbf', 'poly', 'sigmoid'),
                  'C': np.logspace(-4, 4, 9, base=2),
                  'gamma': np.logspace(-6, 0, 7, base=2),
                  'probability': [True],
                  'random_state': [seed]}

dt_param = {'max_features' : ('sqrt', 'log2'),
            'max_depth' : np.arange(1, 31),
            'min_samples_split' : np.arange(2, 21)}

rf_param = {'n_estimators' : [200],
            'max_features' : ('sqrt', 'log2'),
            'max_depth' : np.arange(1, 31),
            'min_samples_split' : np.arange(2, 21),
            'n_jobs': [-1],
            'random_state': [seed]}

extra_param = {'n_estimators': [200],
               'max_features': ('sqrt', 'log2'),
               'max_depth': np.arange(2, 20),
               'min_samples_split': np.arange(2, 20),
               'n_jobs': [-1],
               'random_state': [seed]}

ada_param = {'n_estimators': 20 * np.arange(1, 21),
             'learning_rate': np.logspace(-8, -1, 8, base=2),
             'random_state': [seed]}

xgb_param = {'max_depth': [500],
             'learning_rate': np.logspace(-8, -1, 8, base=2),
             'n_estimators': 20 * np.arange(1, 21),
             'min_child_weight': np.logspace(0, 5, 6, base=2),
             'gamma': np.logspace(0, 5, 6, base=2),
             'reg_lambda': np.logspace(-5, 1, 7),
             'subsample': 0.1 * np.arange(6, 11),
             'colsample_bytree': 0.1 * np.arange(6, 11),
             'seed': [1121]}

knn_param = {'n_neighbors': np.arange(1, 31),
             'weights': ('uniform', 'distance'), 
             'p': (1, 2)}

mlp_param = {'hidden_layer_sizes': [(100, ), (50, ), (50, 50, )],
             'alpha': np.logspace(-5, -1, 5),
             'random_state': [seed]
            }