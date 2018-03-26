import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import ParameterGrid, GridSearchCV, RandomizedSearchCV, StratifiedKFold

def detect_outliers(df, threshold, features):
    outliers_idx = []

    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1

        step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - step) | (df[col] > Q3 + step)].index

        outliers_idx.extend(outlier_list_col)

    outliers_idx = Counter(outliers_idx)
    to_drop = list(idx for idx, outlier_feature in outliers_idx.items()
                        if outlier_feature >= threshold)

    return to_drop

def fill_Nan_with_group_median(df, x, features):
    index = df[x].isnull()
    total_median = df[x].median()
    length = len(df)
    for row in df[index]:
        mask = pd.Series(np.ones(length, dtype=bool))
        for col in features:
            mask = mask & (df[col] == row[col])
        group_median = df.loc[mask, x].median()
        if np.isnan(group_median):
            row[x] = group_median
        else:
            row[x] = total_median


def bar_with_bin(x, y, data, bins=20, **kws):
    _, division = np.histogram(data[y].dropna(), bins=bins)
    labels = ['[' + '%.1f' % division[i] + ', ' + '%.1f' % division[i + 1] + ')'
                for i in range(len(division) - 1)]
    data[y + ' Group'] = pd.cut(data[y], bins=division, labels=labels)
    g = sns.factorplot(x, y + ' Group', order=labels, data=data, orient='h', **kws)
    data.drop(y + ' Group', axis=1, inplace=True)


def onehot_encoding(df, features):
    for col in features:
        df[col] = df[col].astype('category')
        df = pd.get_dummies(df, columns=[col], prefix=col)

    return df

def encode_as_band(df, features, ways=None, bins=5):
    for col in features:
        if ways == 'qcut':
            band = pd.qcut(df[col], q=bins)
        else:
            band = pd.cut(df[col], bins)
        df[col] = band.cat.codes

    return df

def get_train_test_set(file_dir, y_label):
    train_df = pd.read_csv(file_dir + 'train.csv')
    test_df = pd.read_csv(file_dir + 'test.csv')
    X_train = np.array(train_df.drop(y_label, axis=1))
    y_train = np.array(train_df[y_label])
    X_test = np.array(test_df)
    
    return X_train, y_train, X_test

def get_best_classifier(X, y, model, param, n_iter=100, n_folds=5, verbose=True, random_state=528):
    
    if(len(ParameterGrid(param)) >= n_iter):
        g = RandomizedSearchCV(model, param, n_iter=n_iter,
    	                       cv=StratifiedKFold(n_folds, shuffle=True, random_state=random_state),
    	                       verbose=0, n_jobs=n_folds)
        
    else:
        g = GridSearchCV(model, param,
                        cv=StratifiedKFold(n_folds, shuffle=True, random_state=random_state),
                        verbose=0, n_jobs=n_folds)

    g.fit(X, y)
    if(verbose):
        print("=========================================================================")
        print('## ' + model.__class__.__name__ + '(**' + str(g.best_params_) + ')')
        print('score: ', g.best_score_)
        print()
    return g.best_estimator_
