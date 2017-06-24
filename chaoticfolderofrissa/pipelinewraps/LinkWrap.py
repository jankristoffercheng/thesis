from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd
import numpy as np

from features.Links import Links
from utility.DataCleaner import DataCleaner


class LinkWrap(TransformerMixin):

    def __init__(self, target=None):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer()

    def fit(self, X, *args, **kwargs):
        links = []
        for index, row in X.iteritems():
            text = DataCleaner().clean_data(row)
            links.append(Links().get_list_keywords(text))
        dtm  = self.vectorizer.fit_transform(links)
        self.tfidf_transformer.fit(dtm)

        return self

    def transform(self, X, y=None, **transform_params):
        links = []
        for index, row in X.iteritems():
            text = DataCleaner().clean_data(row)
            links.append(Links().get_list_keywords(text))
        dtm  = self.vectorizer.transform(links)
        data=self.tfidf_transformer.transform(dtm)
        print(type(data))
        df = pd.SparseDataFrame(data=[pd.SparseSeries(data[i].toarray().ravel())
                                       for i in np.arange(data.shape[0])],
                                 columns=["Lnk."+freq for freq in self.vectorizer.get_feature_names()])
        # print("1")
        #df=DataFrame(data=data.todense(), columns=list(X.columns.values))
        return df