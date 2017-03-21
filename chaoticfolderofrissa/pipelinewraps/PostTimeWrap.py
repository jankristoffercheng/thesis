from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class PostTimeWrap(TransformerMixin):

    def transform(self, X, **transform_params):
        hours = X.apply(enrange)

        mlb = MultiLabelBinarizer(classes=[0,1,2,3,4])
        temp = hours.apply(lambda x: [x])
        data = DataFrame(data=mlb.fit_transform(temp), columns=["Tim."+time for time in getClasses()])
        return data

    def fit(self, X, y=None, **fit_params):
        return self

def getClasses():
    return ['0-4','5-9','10-14','15-19','20-23']

def enrange(x):
    if(x>=0 and x<=4): return 0
    elif(x>=5 and x<=9): return 1
    elif(x>=10 and x<=14): return 2
    elif(x>=15 and x<=19): return 3
    elif(x>=20 and x<=23): return 4

    print("ina")
    return -1