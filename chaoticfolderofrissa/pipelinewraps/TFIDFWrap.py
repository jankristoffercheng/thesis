from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np

class TFIDFWrap(TransformerMixin):

    def __init__(self, target=None):
        self.tfidf_transformer = TfidfTransformer()

    def fit(self, X, *args, **kwargs):
        self.tfidf_transformer.fit(X.values)

        return self

    def transform(self, X, y=None, **transform_params):
        data=self.tfidf_transformer.transform(X.values)
        print(type(data))
        df = pd.SparseDataFrame(data=[pd.SparseSeries(data[i].toarray().ravel())
                                       for i in np.arange(data.shape[0])],
                                 columns=list(X.columns.values))
        print("1")
        #df=DataFrame(data=data.todense(), columns=list(X.columns.values))
        return df