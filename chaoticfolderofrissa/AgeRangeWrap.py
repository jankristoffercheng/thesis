from pandas import DataFrame
from sklearn.base import TransformerMixin


class AgeRangeWrap(TransformerMixin):

    def transform(self, X, **transform_params):

        agerange = DataFrame(X.apply(enrange))
        return agerange

    def fit(self, X, y=None, **fit_params):
        return self


def enrange(x):
    if(x>=18 and x<=24):
        return 0
    elif(x>=25 and x<=34):
        return 1
    elif(x>=34 and x<=44): return 2
    elif(x>-45 and x<=54): return 3
    elif(x>=55 and x<=64): return 4

    return -1