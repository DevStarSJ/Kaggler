import pandas as pd


class CVFileMixer:

    @staticmethod
    def get(df_list, validation_index):
        train = pd.concat([df_list[i] for i in range(len(df_list)) if i != validation_index])
        validation = df_list[validation_index]

        return train, validation
