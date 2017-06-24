from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class StackAgeRangeWrap(TransformerMixin):

    def transform(self, X, **transform_params):

        # agerange = X.apply(enrange)

        mlb = MultiLabelBinarizer(classes=[0,1,2,3,4])
        temp = X.apply(lambda x: [x])
        data = DataFrame(data=mlb.fit_transform(temp), columns=["Stk." + age for age in getClasses()], index=X.index.values)

        return data

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    return ['18-24','25-34','34-44','45-54','55-64']

def enrange(x):
    if(x>=18 and x<=24):
        return 0
    elif(x>=25 and x<=34):
        return 1
    elif(x>=34 and x<=44): return 2
    elif(x>-45 and x<=54): return 3
    elif(x>=55 and x<=65): return 4

    return -1