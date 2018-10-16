from copy import deepcopy
import numpy as np
import pandas as pd

class CrossValidationModel:
    def __init__(self):
        self.n_splits = None
        self.cv_train_Xs = []
        self.cv_train_ys = []
        self.cv_test_Xs = []
        self.cv_test_ys = []
        self.cv_test_predictions = []
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
            
            cv_model = deepcopy(model)
            cv_model.fit(train_X, train_y)
            
            test_predict = cv_model.predict(test_X)
            
            self.cv_train_Xs.append(train_X)
            self.cv_train_ys.append(train_y)
            self.cv_test_Xs.append(test_X)
            self.cv_test_ys.append(test_y)
            self.cv_test_predictions.append(test_predict)
            self.models.append(cv_model)
            
    def score(self, metric):
        return [metric(self.cv_test_predictions[i], self.cv_test_ys[i]) for i in range(self.n_splits)]
    
    def predict(self, data):
        return [self.models[i].predict(data) for i in range(self.n_splits)]
    
    def predict_stack(self, data):
        predicts = self.predict(data)
        return pd.DataFrame({str(i) : v for i, v in enumerate(predicts)})
    
    def predict_average(self, data):
        predicts = self.predict(data)
        
        len_cv = len(predicts)
        len_data = len(predicts[0])
            
        return [sum([predicts[j][i] for j in range(len_cv)]) / len_cv for i in range(len_data)]