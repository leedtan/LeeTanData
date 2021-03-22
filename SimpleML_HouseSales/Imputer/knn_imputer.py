import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer as Imputer



class KNNImputer:
    def __init__(self):
        self.regressors = {}

    @staticmethod
    def train_one_regreessor(df, df_full_data, col):
        input_features = df_full_data[[c for c in df.columns if c != col]]
        output_features = df_full_data[col]
        regressor = KNeighborsRegressor()
        regressor.fit(input_features, output_features)
        return regressor

    def fit(self, df):
        self.imputer = Imputer().fit(df)
        df_full_data = df.dropna(axis=0)
        for col in df.columns:
            self.regressors[col] = self.train_one_regreessor(
                df, df_full_data, col)

    def fill_in(self, df):
        dfnull = pd.isnull(df)
        df_filled_in = pd.DataFrame(
            self.imputer.transform(df), columns=df.columns)

        for col in df.columns:
            cols = [c for c in df.columns if c != col]
            input_features = df_filled_in[cols]
            output_missing = dfnull[col]
            if output_missing.sum() > 0:
                regressor = self.regressors[col]
                df.loc[output_missing, col] = regressor.predict(
                    input_features[output_missing.values])
