from sklearn.base import TransformerMixin


class AgeRangeWrap(TransformerMixin):
    """
    Transforms the age to numerical labels
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """

    def transform(self, X, **transform_params):

        agerange = X.apply(enrange)

        return agerange

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    """
    :return: array of the age ranges
    """
    return ['18-24','25-34','34-44','45-54','55-64']
    # return ['18-24','25-64']

def enrange(x):
    """
    :param x: age of the user
    :return: age range group
    """
    if(x>=18 and x<=24):
        return 0
    elif(x>=25 and x<=34):
        return 1
    elif(x>=34 and x<=44): return 2
    elif(x>-45 and x<=54): return 3
    elif(x>=55 and x<=65): return 4

    return -1