import pandas as pd
from sklearn.base import TransformerMixin

from features.Structure import Structure
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
    """
    Processes all structure features of the data.
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """
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