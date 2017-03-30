from pandas import DataFrame
from sklearn.base import TransformerMixin
import numpy as np

class ExtractionWrap(TransformerMixin):

    def __init__(self, extraction, target=None):
        self.extraction = extraction
        self.target=target


    def fit(self, X, *args, **kwargs):
        self.extraction = self.extraction.fit(X, self.target)

        return self

    def transform(self, X, y=None, **transform_params):
        data=self.extraction.transform(X)
        result = DataFrame(data=data)
        return result