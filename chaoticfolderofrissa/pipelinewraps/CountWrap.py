from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class CountWrap(TransformerMixin):

    def __init__(self, target=None):
        self.vectorizer = CountVectorizer()

    def fit(self, X, *args, **kwargs):
        self.vectorizer.fit_transform(X)

        return self

    def transform(self, X, y=None, **transform_params):
        data = self.vectorizer.transform(X)
        dtm = DataFrame(data=data.todense(), columns=["Frq."+freq for freq in self.vectorizer.get_feature_names()])
        return dtm