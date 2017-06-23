from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd
import numpy as np

from features.Context import Context
from features.EmojisEmoticons import EmojisEmoticons
from features.Links import Links
from utility.DataCleaner import DataCleaner


class EmojiWrap(TransformerMixin):

    def __init__(self, target=None):
        self.emoji = EmojisEmoticons()

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X, y=None, **transform_params):

        data = self.emoji.getEmojiTFIDF(X)
        df = pd.DataFrame(data=data.todense(),
                                 columns=["Soc."+freq for freq in self.emoji.getLabels()])
        print("1")
        #df=DataFrame(data=data.todense(), columns=list(X.columns.values))
        print(df.shape)
        return df