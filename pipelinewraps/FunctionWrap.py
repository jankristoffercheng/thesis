import pandas as pd
from sklearn.base import TransformerMixin

from features.FunctionWordCount import FunctionWordCount
from utility.DataCleaner import DataCleaner

columnList = ('article',
              'prosentence',
              'pronouns',
              'auxillary',
              'conjunction',
              'interjection',
              'adposition')

class FunctionWrap(TransformerMixin):
    """
    Processes all function word features of the data.
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """
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
        # print(result.shape)
        return result