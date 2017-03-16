from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


class VectorizeWrap(TransformerMixin):
    def transform(self, X, **transform_params):
        mlb = MultiLabelBinarizer()
        temp = X.apply(lambda x: [x])
        data = DataFrame( mlb.fit_transform(temp))
        print("Vectorize", data.shape)
        return data

    def fit(self, X, y=None, **fit_params):
        return self