import pandas as pd


class CVFileMixer:

    @staticmethod
    def get_cv_train_validation(df_list, validation_index):
        train = pd.concat([df_list[i] for i in range(len(df_list)) if i != validation_index])
        validation = df_list[validation_index]

        return train, validation
    
    @staticmethod
    def get_predict_stacked(df_list, value_column):
        return pd.DataFrame({str(i) : v[value_column] for i, v in enumerate(df_list)})
    
    @staticmethod
    def get_predict_mean(df_list, value_column):
        predicts = [v[value_column] for v in df_list]
        
        len_cv = len(predicts)
        len_data = len(predicts[0])
            
        return [sum([predicts[j][i] for j in range(len_cv)]) / len_cv for i in range(len_data)]
        
