import uuid
from re import findall
from sklearn.base import TransformerMixin

import pandas as pd
from features.POSSequencePattern import POSSequencePattern
from features.Receptiviti import Receptiviti
from features.Structure import Structure
from model.Document import Document
from utility.DataCleaner import DataCleaner

columnList = ('sentenceCount',
              'paragraphCount',
              'AvgWordPerPar',
              'AvgCharPerPar',
              'AvgSentPerPar',
              'AvgWordPerSent',
              'UpperSentenceCount',
              'LowerSentenceCount')

class StructureWrap(TransformerMixin):
    def __init__(self):
        self.feature= Structure()
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        result = pd.DataFrame(columns=['Str.'+item for item in columnList], dtype='float')
        for index, row in X.iteritems():
            # print(row)
            row = DataCleaner().clean_data(row)
            data = []
            data.append(self.feature.getNSentences(row))
            data.append(self.feature.getNParagraphs(row))
            data.append(self.feature.getAvgNWordPerParagraph(row))
            data.append(self.feature.getAvgNCharacterPerParagraph(row))
            data.append(self.feature.getAvgNSentencePerParagraph(row))
            data.append(self.feature.getAvgNWordPerSentence(row))
            data.append(self.feature.getNSentenceBegUpper(row))
            data.append(self.feature.getNSentenceBegLower(row))
            result.loc[index] = data

        return result