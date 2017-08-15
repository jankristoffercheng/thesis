from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class StackAgeRangeWrap(TransformerMixin):
    """
    Transforms the age multiclass to multilabel binary
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """

    def transform(self, X, **transform_params):

        # agerange = X.apply(enrange)

        mlb = MultiLabelBinarizer(classes=[0,1])
        temp = X.apply(lambda x: [x])
        data = DataFrame(data=mlb.fit_transform(temp), columns=["Stk." + age for age in getClasses()], index=X.index.values)

        return data

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    """
    :return: array of the age ranges
    """
    return ['18-24','25-34','34-44','45-54','55-64']
    # return ['18-24','25-64']
