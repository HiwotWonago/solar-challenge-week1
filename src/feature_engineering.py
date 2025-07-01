import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AggregateTransactionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='CustomerId', amount_col='Amount'):
        self.id_col = id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.id_col)[self.amount_col].agg(
            total_amount='sum',
            avg_amount='mean',
            transaction_count='count',
            std_amount='std'
        ).reset_index()
        return agg

class TransactionTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors='coerce')

        df['TransactionHour'] = df[self.datetime_col].dt.hour
        df['TransactionDay'] = df[self.datetime_col].dt.day
        df['TransactionMonth'] = df[self.datetime_col].dt.month
        df['TransactionYear'] = df[self.datetime_col].dt.year

        return df


