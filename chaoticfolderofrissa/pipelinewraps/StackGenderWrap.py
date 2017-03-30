from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class StackGenderWrap(TransformerMixin):

    def transform(self, X, **transform_params):

        # gen = X.apply(enrange)

        mlb = MultiLabelBinarizer(classes=[0,1])
        temp = X.apply(lambda x: [x])
        data = DataFrame(data=mlb.fit_transform(temp), columns=["Stk." + gender for gender in getClasses()])

        return data

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    return ['F','M']

def enrange(x):
    if(x=='F'):
        return 0
    else:
        return 1
