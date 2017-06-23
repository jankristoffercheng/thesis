from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd
import numpy as np

from features.Context import Context
from features.Links import Links
from utility.DataCleaner import DataCleaner


class ContextualWrap(TransformerMixin):

    def __init__(self, target=None):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer()

    def fit(self, X, *args, **kwargs):
        data = []
        for index, row in X.iteritems():
            text = DataCleaner().clean_data(row)
            data.append(Context().process(text))
        dtm  = self.vectorizer.fit_transform(data)
        self.tfidf_transformer.fit(dtm)

        return self

    def transform(self, X, y=None, **transform_params):
        data = []
        for index, row in X.iteritems():
            text = DataCleaner().clean_data(row)
            data.append(Context().process(text))
        dtm  = self.vectorizer.transform(data)
        data=self.tfidf_transformer.transform(dtm)
        print(type(data))
        df = pd.DataFrame(data=data.todense(),
                                 columns=["Soc."+freq for freq in self.vectorizer.get_feature_names()])
        print(df.shape)
        #df=DataFrame(data=data.todense(), columns=list(X.columns.values))
        return df