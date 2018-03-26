import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

class StackingClassifier(object):
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.col_name = []
        self.cv_error = []

    def add_prediction(self, X, y, T, model, name):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        new_train = model.predict(X)
        new_test = model.predict(T)
        if len(self.col_name) > 0:
            self.S_train = np.hstack([self.S_train, new_train])
            self.S_test = np.hstack([self.S_test, new_test])
        else:
            self.S_train = new_train
            self.S_test = new_test

        self.col_name.append(name)
        pass

    def get_uniform_blend(self, best_n):
        pass


    def add_predictionCV(self, X, y, T, model, name, random_state=528, log_proba=False, n_classes=2):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=random_state)
        if(log_proba):
            new_train = np.zeros((X.shape[0], n_classes - 1))
            new_test = np.zeros((T.shape[0], n_classes - 1))
        else:
            new_train = np.zeros(X.shape[0])
            new_test = np.zeros(T.shape[0])

        for train_idx, test_idx in folds.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_holdout = X[test_idx]
            model.fit(X_train, y_train)
            if log_proba:
                new_train[test_idx, :] = model.predict_proba(X_holdout)[:, 1:]
                new_test = new_test + model.predict_proba(T)[:, 1:]
            else:
                new_train[test_idx] = model.predict(X_holdout)
                new_test = new_test + model.predict(T)

        self.cv_error.append(np.equal(new_train, y).mean())
        # Majority vote on prediction on test by models trained by different train folds
        if log_proba:
            new_test = (new_test / self.n_folds)
        else:
            new_train = new_train.reshape(new_train.shape[0], 1)
            new_test = (new_test > (self.n_folds / 2)).astype(int).reshape(new_test.shape[0], 1)

        if len(self.col_name) > 0:
            self.S_train = np.hstack([self.S_train, new_train])
            self.S_test = np.hstack([self.S_test, new_test])
        else:
            self.S_train = new_train
            self.S_test = new_test

        self.col_name.append(name)


    def get_train_test_set(self):
        return self.S_train, self.S_test

    def get_CVerror(self):
        return {name: error for (name, error) in zip(self.col_name, self.cv_error)}

    def plot_corr_heatmap(self):
        if(self.S_train.shape[1] == len(self.col_name)):
            sns.heatmap(pd.DataFrame(self.S_train, columns=self.col_name).corr(), annot=True)
        else:
            sns.heatmap(pd.DataFrame(self.S_train).corr(), annot=True)
