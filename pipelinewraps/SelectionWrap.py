import numpy as np
from pandas import DataFrame
from sklearn.base import TransformerMixin


class SelectionWrap(TransformerMixin):
    """
    Performs feature selection
    """
    def __init__(self, selection, target=None):
        """
        :param selection: feature selection technique
        :param target: true classes of the data
        """
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