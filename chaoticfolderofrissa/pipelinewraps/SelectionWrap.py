from pandas import DataFrame
from sklearn.base import TransformerMixin
import numpy as np

class SelectionWrap(TransformerMixin):

    def __init__(self, selection, target=None):
        self.selection = selection
        self.target=target


    def fit(self, X, y, *args, **kwargs):
        if(self.target is None):
            self.selection = self.selection.fit(X, y)
        else:
            self.selection = self.selection.fit(X, self.target)

        return self

    def transform(self, X, y=None, **transform_params):
        data=self.selection.transform(X)
        col= np.asarray(list(X.columns.values))[self.selection.get_support()]
        result = DataFrame(data=data, columns=col)
        return result