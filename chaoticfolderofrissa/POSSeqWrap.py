from sklearn.base import TransformerMixin

import pandas as pd
from features.POSSequencePattern import POSSequencePattern
from model.Document import Document


class POSSeqWrap(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = X[['Text','POS']]
        p = POSSequencePattern(dfToDocument(df))
        top = p.minePOSPatterns(0.3,0.2)

        result = pd.DataFrame(columns=range(len(top)), dtype='float')
        for index, row in df.iterrows():
            countlist = []
            for pattern in top:
                count = row['Text'].count(pattern)
                countlist.append(count)
            result.loc[index] = countlist

        return result

def dfToDocument(df):
    list = []
    for index, row in df.iterrows():
        list.append(Document(row['Text'], row['POS']))
    return list