from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class GenderWrap(TransformerMixin):

    def transform(self, X, **transform_params):

        gen = X.apply(enrange)

        return gen

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    return ['F','M']

def enrange(x):
    if(x=='F'):
        return 0
    else:
        return 1