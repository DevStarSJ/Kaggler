"""
Faron의 stacking-starter code의 일부를 사용함
https://www.kaggle.com/mmueller/stacking-starter

stacking util
"""
# utility
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# algorithm
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import ElasticNetCV, LassoLarsCV

# metric
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error, r2_score


class SklearnWrapper(object):
    def __init__(self, clf, params=None, **kwargs):
        seed = kwargs.get('seed', 0)
        y_value_log = kwargs.get('y_value_log', False)

        params['random_state'] = seed
        self.clf = clf(**params)
        self.y_value_log = y_value_log

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        if self.y_value_log is True:
            self.clf.fit(x_train, np.log(y_train))
        else:
            self.clf.fit(x_train, y_train)

    def predict(self, x):
        if self.y_value_log is True:
            return np.exp(self.clf.predict(x))
        else:
            return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, params=None, **kwargs):
        seed = kwargs.get('seed', 0)
        num_rounds = kwargs.get('num_rounds', 1000)
        early_stopping = kwargs.get('ealry_stopping', 100)
        eval_function = kwargs.get('eval_function', None)
        verbose_eval = kwargs.get('verbose_eval', 100)
        y_value_log = kwargs.get('y_value_log', False)
        base_score = kwargs.get('base_score', False)
        feval_maximize = kwargs.get('maximize', False)

        if 'silent' not in params:
            params['silent'] = 1

        self.param = params
        self.param['seed'] = seed
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping

        self.eval_function = eval_function
        self.verbose_eval = verbose_eval
        self.y_value_log = y_value_log
        self.base_score = base_score
        self.feval_maximize = feval_maximize

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        need_cross_validation = True
        if x_cross is None:
            need_cross_validation = False

        if self.base_score is True:
            y_mean = np.mean(y_train)
            self.param['base_score'] = y_mean

        if self.y_value_log is True:
            y_train = np.log(y_train+1)
            if need_cross_validation is True:
                y_cross = np.log(y_cross+1)

        if need_cross_validation is True:
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dvalid = xgb.DMatrix(x_cross, label=y_cross)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

            self.clf = xgb.train(self.param, dtrain, self.num_rounds, watchlist, feval=self.eval_function,
                                 early_stopping_rounds=self.early_stopping, maximize=self.feval_maximize,
                                 verbose_eval=self.verbose_eval)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train, silent= True)
            self.clf = xgb.train(self.param, dtrain, self.num_rounds)

    def predict(self, x):
        if self.y_value_log is True:
            return np.exp(self.clf.predict(xgb.DMatrix(x)))-1
        else:
            return self.clf.predict(xgb.DMatrix(x))

    def get_params(self):
        return self.param


class LgbmWrapper(object):
    def __init__(self, params=None, **kwargs):
        seed = kwargs.get('seed', 0)
        num_rounds = kwargs.get('num_rounds', 1000)
        early_stopping = kwargs.get('ealry_stopping', 100)
        eval_function = kwargs.get('eval_function', None)
        verbose_eval = kwargs.get('verbose_eval', 100)
        y_value_log = kwargs.get('y_value_log', False)
        base_score = kwargs.get('base_score', False)
        feval_maximize = kwargs.get('maximize', False)

        self.param = params
        self.param['seed'] = seed
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping

        self.eval_function = eval_function
        self.verbose_eval = verbose_eval
        self.y_value_log = y_value_log
        self.base_score = base_score
        self.feval_maximize = feval_maximize

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        need_cross_validation = True
        if x_cross is None:
            need_cross_validation = False

        if isinstance(y_train, pd.DataFrame) is True:
            y_train = y_train[y_train.columns[0]]
            if need_cross_validation is True:
                y_cross = y_cross[y_cross.columns[0]]

        if self.base_score is True:
            y_mean = np.mean(y_train)
            self.param['init_score '] = y_mean

        if self.y_value_log is True:
            y_train = np.log(y_train+1)
            if need_cross_validation is True:
                y_cross = np.log(y_cross+1)

        if need_cross_validation is True:
            dtrain = lgbm.Dataset(x_train, label=y_train, silent=True)
            dvalid = lgbm.Dataset(x_cross, label=y_cross, silent=True)
            self.clf = lgbm.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=dvalid,
                                  feval=self.eval_function, early_stopping_rounds=self.early_stopping,
                                  verbose_eval=self.verbose_eval)
        else:
            dtrain = lgbm.Dataset(x_train, label=y_train, silent= True)
            self.clf = lgbm.train(self.param, dtrain, self.num_rounds)

    def predict(self, x):
        if self.y_value_log is True:
            return np.exp(self.clf.predict(x, num_iteration=self.clf.best_iteration))-1
        else:
            return self.clf.predict(x, num_iteration=self.clf.best_iteration)

    def get_params(self):
        return self.param


class KerasWrapper(object):
    def __init__(self, model, callback, **kwargs):
        self.model = model
        self.callback = callback

        epochs = kwargs.get('epochs', 300)
        batch_size = kwargs.get('batch_size', 30)
        shuffle = kwargs.get('shuffle', True)
        verbose_eval = kwargs.get('verbose_eval',1)

        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose_eval = verbose_eval

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        need_cross_validation = True

        if x_cross is None:
            need_cross_validation = False

        if isinstance(y_train, pd.DataFrame) is True:
            y_train = y_train[y_train.columns[0]]
            if need_cross_validation is True:
                y_cross = y_cross[y_cross.columns[0]]

        if isinstance(x_train, pd.DataFrame) is True:
            x_train = x_train.values
            if need_cross_validation is True:
                x_cross = x_cross.values

        if need_cross_validation is True:
            self.history = self.model.fit(x_train, y_train,
                                          nb_epoch=self.epochs,
                                          batch_size = self.batch_size,
                                          validation_data=(x_cross, y_cross),
                                          verbose=self.verbose_eval,
                                          callbacks=self.callback,
                                          shuffle=self.shuffle)
        else:
            self.model.fit(x_train, y_train,
                           nb_epoch=self.epochs,
                           batch_size=self.batch_size,
                           verbose=self.verbose_eval,
                           callbacks=self.callback,
                           shuffle=self.shuffle)

    def predict(self, x):
        print(x.shape)
        if isinstance(x, pd.DataFrame) is True:
            x = x.values

        result = self.model.predict(x)
        return result.flatten()

    def get_params(self):
        return self.model.summary()


def get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs):
    nfolds = kwargs.get('n_folds', 5)
    kfold_shuffle = kwargs.get('kfold_shuffle', True)
    kfold_random_state = kwargs.get('kfold_random_sate', 0)
    y_value_log = kwargs.get('y_value_log', False)


    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    kf = KFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((nfolds, ntest))
    if y_value_log is True:
        y_train = np.log(y_train+1)
    cv_sum = 0
    try:
        if clf.clf is not None:
            print(clf.clf)
    except:
        print(clf)
        print(clf.get_params())

    for i, (train_index, cross_index) in enumerate(kf.split(x_train)):
        x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]
        y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]

        clf.train(x_tr, y_tr, x_cr, y_cr)

        oof_train[cross_index] = clf.predict(x_cr)

        cv_score = eval_func(y_cr, oof_train[cross_index])

        print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)
        cv_sum = cv_sum + cv_score

        if y_value_log is True:
            oof_test_skf[i, :] = np.exp(clf.predict(x_test))-1
        else:
            oof_test_skf[i, :] = clf.predict(x_test)

    score = cv_sum / nfolds
    print("Average CV-Score: ", score)
    if y_value_log is True:
        oof_train = np.exp(oof_train)-1

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def kfold_test(clf, x_train, y_train, eval_func, **kwargs):
    nfolds = kwargs.get('NFOLDS', 5)
    kfold_shuffle = kwargs.get('kfold_shuffle', True)
    kfold_random_state = kwargs.get('kfold_random_sate', 0)
    y_value_log = kwargs.get('y_value_log', False)

    ntrain = x_train.shape[0]

    kf = KFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

    if y_value_log is True:
        y_train = np.log(y_train+1)
    cv_sum = 0
    try:
        if clf.clf is not None:
            print(clf.clf)
    except:
        print(clf)
        print(clf.get_params())

    best_rounds = []
    for i, (train_index, cross_index) in enumerate(kf.split(x_train)):
        x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]
        y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]

        clf.train(x_tr, y_tr, x_cr, y_cr)

        cv_score = eval_func(y_cr, clf.predict(x_cr))

        print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)
        cv_sum = cv_sum + cv_score
        best_rounds.append(clf.clf.best_iteration)

    score = cv_sum / nfolds
    print("Average CV-Score: ", score)

    return score, np.max(best_rounds)


if __name__ == '__main__':
    print('test code')
    x_train = pd.read_csv('input/default_x_train.csv')
    y_train = pd.read_csv('input/default_y_train.csv')
    x_test = pd.read_csv('input/default_x_test.csv')

    lgbm_params = {
        'boosting_type': 'gbdt', 'objective': 'regression',
        'learning_rate': 0.01, 'subsample': 0.8, 'max_depth': 5,
        'metric': 'mae'
    }

    lgbm_model = LgbmWrapper(params=lgbm_params, num_rounds=3000, ealry_stopping=100,
                                      eval_function=mean_absolute_error,
                                      verbose_eval=30, base_score=True, maximize=False, y_value_log=False)

    lgbm_train_oof, lgbm_test_oof = get_oof(lgbm_model, x_train.fillna(0), y_train, x_test.fillna(0),
                                                     mean_absolute_error, NFOLDS=5)
