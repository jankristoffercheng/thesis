from sklearn.base import TransformerMixin


class GenderWrap(TransformerMixin):
    """
    Transforms the gender to numerical labels
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """

    def transform(self, X, **transform_params):

        gen = X.apply(enrange)

        return gen

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    """
    :return: returns the gender classes
    """
    return ['F','M']

def enrange(x):
    """
    :param x: gender of the user
    :return: 0 for F and 1 for M
    """
    if(x=='F'):
        return 0
    else:
        return 1
