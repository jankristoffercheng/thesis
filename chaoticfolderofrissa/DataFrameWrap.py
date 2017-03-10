from pandas import DataFrame
from sklearn.base import TransformerMixin


class DataFrameWrap(TransformerMixin):

    def transform(self, X, **transform_params):
        data = DataFrame(X)
        return data

    def fit(self, X, y=None, **fit_params):
        return self