import uuid
from re import findall
from sklearn.base import TransformerMixin

import pandas as pd

from features.CharacterFeatures import CharacterFeatures
from features.POSSequencePattern import POSSequencePattern
from features.Receptiviti import Receptiviti
from features.functionwords.FunctionWordCount import FunctionWordCount
from model.Document import Document
from utility.DataCleaner import DataCleaner

columnList = ('article',
              'prosentence',
              'pronouns',
              'auxillary',
              'conjunction',
              'interjection',
              'adposition')

class FunctionWrap(TransformerMixin):
    def __init__(self):
        self.features = FunctionWordCount()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        result = pd.DataFrame(columns=['Soc.'+item for item in columnList], dtype='float')
        for index, row in X.iteritems():
            print(row)
            row = DataCleaner().clean_data(row)
            data = []
            data.append(self.features.getArticleCount(row))
            data.append(self.features.getProSentenceCount(row))
            data.append(self.features.getPronounCount(row))
            data.append(self.features.getAuxillaryCount(row))
            data.append(self.features.getConjunctionCount(row))
            data.append(self.features.getInterjectionCount(row))
            data.append(self.features.getAdpositionCount(row))

            result.loc[index] = data
        print(result.shape)
        return result