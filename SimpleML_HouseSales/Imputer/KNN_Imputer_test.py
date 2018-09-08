import numpy as np
import pandas as pd
from knn_imputer import KNNImputer
from sklearn.preprocessing import Imputer

if __name__ == '__main__':
    # df = pd.read_csv(
    #     'ListingsAndSales.csv', parse_dates=['ListingDate', 'SalesDate'])
    # df = df.select_dtypes(include=[np.number])
    num_rows = 200
    df = pd.DataFrame({
        'a':
        np.arange(num_rows) + np.random.rand(num_rows),
        'b':
        np.arange(num_rows, 0, -1) + np.random.rand(num_rows),
        'c':
        np.sin(np.arange(num_rows) / 100 + np.random.rand(num_rows) / 100),
        'c':
        np.cos(np.arange(num_rows) / 100 + np.random.rand(num_rows) / 100),
    })
    remove_cols = np.random.choice(df.columns, size=100)
    remove_rows = np.random.choice(num_rows, size=100)
    df_raw = df.copy()
    for idx, row in enumerate(remove_rows):
        col = remove_cols[idx]
        df.loc[row, col] = np.nan
    df_knn = df.copy()
    df_sklearn = df.copy()
    #df_knn.loc[remove_rows, remove_cols] = np.nan
    imputer = KNNImputer()
    imputer.fit(df_knn)
    #print(df_knn.isnull().sum())
    imputer.fill_in(df_knn)
    assert (df_knn.isnull().sum().sum() == 0)
    #print(df_knn.isnull().sum())
    sklearnImputer = Imputer().fit(df_sklearn)
    df_sklearn = sklearnImputer.transform(df_sklearn)
    err_sklearn = (df_sklearn - df_raw).values**2
    print(err_sklearn.sum())
    err_knn = (df_knn - df_raw).values**2
    print(err_knn.sum())
    assert (err_knn.sum() < (err_sklearn.sum() / 3))
    print('df knn much less than df sklearn!')
