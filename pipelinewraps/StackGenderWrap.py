from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class StackGenderWrap(TransformerMixin):
    """
    Transforms the gender multiclass to multilabel binary
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """

    def transform(self, X, **transform_params):

        # gen = X.apply(enrange)

        mlb = MultiLabelBinarizer(classes=[0,1])
        temp = X.apply(lambda x: [x])
        data = DataFrame(data=mlb.fit_transform(temp), columns=["Stk." + gender for gender in getClasses()], index=X.index.values)

        return data

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    """
    :return: returns the gender classes
    """
    return ['F','M']
