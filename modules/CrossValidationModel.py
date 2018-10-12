from copy import deepcopy
import numpy as np
import pandas as pd

class CrossValidationModel:
    def __init__(self):
        self.n_splits = None
        self.train_Xs = []
        self.train_ys = []
        self.test_Xs = []
        self.test_ys = []
        self.test_predictions = []
        self.models = []
        #self.score = []
        pass
    
    def fit(self, model, data, n_splits=10, target_col = 'target', shuffle=True):
        
        if target_col not in data.columns:
            raise ValueError("[target_col] isn't in columns of [data]")
                        
        self.n_splits = n_splits
        
        df = data.copy(deep=True)
        
        if shuffle:
            df = df.sample(n=len(df))
            
        df_list = np.array_split(df, n_splits)
        
        for i in range(n_splits):
            test_set = df_list[i]
            train_set = pd.concat([df_list[j] for j in range(n_splits) if j != i])
            
            train_y = train_set[target_col]
            train_X = train_set.drop(columns=target_col)
            test_y = test_set[target_col]
            test_X = test_set.drop(columns=target_col)
            
            cv_model = model.fit(train_X, train_y)
            test_predict = model.predict(test_X)
            
            self.train_Xs.append(train_X)
            self.train_ys.append(train_y)
            self.test_Xs.append(test_X)
            self.test_ys.append(test_y)
            self.test_predictions.append(test_predict)
            self.models.append(deepcopy(cv_model))
            
    def score(self, metric):
        return [metric(self.test_predictions[i], self.test_ys[i]) for i in range(self.n_splits)]
    
    def predict(self, data):
        return [self.models[i].predict(data) for i in range(self.n_splits)]

# train_set and test_set is DataFrame of Kaggle Titanic Dataset.
# This code is example of CrossValidationModel Usage.    
    
# if __name__ == '__main__':
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.metrics import accuracy_score, roc_auc_score
    
#     cvm = CrossValidationModel()
#     model = RandomForestClassifier(n_estimators=13)
#     n_splits = 10
#     target_col = 'Survived'
    
#     cvm.fit(model, train_set, n_splits, 'Survived')
    
#     print(cvm.score(accuracy_score))
#     print(cvm.score(roc_auc_score))
    
#     predicts = cvm.predict(test_set)
    
#     for i in range(n_splits):
#         print(predicts[i].sum())