import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from features.Links import Links
from utility.DataCleaner import DataCleaner


class LinkWrap(TransformerMixin):
    """
    Processes all link features of the data.
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """

    def __init__(self, target=None):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer(stop_words='english')

    def fit(self, X, *args, **kwargs):
        links = []
        for index, row in X.iteritems():
            text = DataCleaner().clean_email(row)
            links.append(Links().get_list_keywords(text))
        dtm  = self.vectorizer.fit_transform(links)
        self.tfidf_transformer.fit(dtm)

        return self

    def transform(self, X, y=None, **transform_params):
        links = []
        # print(links)
        for index, row in X.iteritems():
            text = DataCleaner().clean_email(row)
            links.append(Links().get_list_keywords(text))
        dtm  = self.vectorizer.transform(links)
        data=self.tfidf_transformer.transform(dtm)
        df = pd.SparseDataFrame(data=[pd.SparseSeries(data[i].toarray().ravel())
                                       for i in np.arange(data.shape[0])],
                                 columns=["Lnk."+freq for freq in self.vectorizer.get_feature_names()])
        # print("1")
        #df=DataFrame(data=data.todense(), columns=list(X.columns.values))
        return df