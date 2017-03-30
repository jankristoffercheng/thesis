from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np

class CountWrap(TransformerMixin):

    def __init__(self, target=None):
        self.vectorizer = CountVectorizer()

    def fit(self, X, *args, **kwargs):
        self.vectorizer.fit_transform(X)

        return self

    def transform(self, X, y=None, **transform_params):
        data = self.vectorizer.transform(X)
        print(type(data))
        print(data.shape, data.shape[0])
        dtm = pd.SparseDataFrame(data = [ pd.SparseSeries(data[i].toarray().ravel())
                              for i in np.arange(data.shape[0]) ], columns=["Frq."+freq for freq in self.vectorizer.get_feature_names()] )
        print("1")
        #dtm = DataFrame(data=data.todense(), columns=["Frq."+freq for freq in self.vectorizer.get_feature_names()])
        return dtm