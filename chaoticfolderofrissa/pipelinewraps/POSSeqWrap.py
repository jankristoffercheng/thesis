from re import findall
from sklearn.base import TransformerMixin

import pandas as pd
from features.POSSequencePattern import POSSequencePattern
from model.Document import Document
from utility.DataCleaner import DataCleaner


class POSSeqWrap(TransformerMixin):
    def __init__(self):
        self.top = None

    def fit(self, X, y=None, **fit_params):
        data = X[['Text', 'POS']]
        p = POSSequencePattern(dfToDocument(data))
        self.top = p.minePOSPatterns(0.3, 0.2)

        return self

    def transform(self, X, y=None, **transform_params):

        data = X[['Text', 'POS']]
        result = pd.DataFrame(columns=["POS."+seq for seq in self.top], dtype='float')
        for index, row in data.iterrows():
            countlist = []
            for pattern in self.top:
                count = len(findall('(?='+pattern+')', row["POS"]))
                countlist.append(count)
            result.loc[index] = countlist

        return result

def dfToDocument(df):
    list = []
    for index, row in df.iterrows():
        list.append(Document(
            DataCleaner().clean_data(row['Text']), row['POS']))
    return list