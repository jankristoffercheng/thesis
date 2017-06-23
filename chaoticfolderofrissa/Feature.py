from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.pipelinewraps.AgeRangeWrap import AgeRangeWrap
from chaoticfolderofrissa.pipelinewraps.ExtractionWrap import ExtractionWrap
from chaoticfolderofrissa.pipelinewraps.GenderWrap import GenderWrap
from chaoticfolderofrissa.pipelinewraps.SelectionWrap import SelectionWrap
import pandas as pd
import numpy as np

class Feature:

    def __init__(self, X, y, source, freqdata):
        self.X,self.y = X, y
        self.y['Gender'] = GenderWrap().fit_transform(self.y['Gender'])
        self.y['Age'] = AgeRangeWrap().fit_transform(self.y['Age'])
        temp = pd.read_csv("data/"+source+"/raw/features_cmp.csv", encoding = "ISO-8859-1")
        self.features = pd.concat([temp,freqdata], axis=1)

    def applySelection(self, selection, type):
        # print(self.y[['Gender', 'Age']])
        # print(self.y['Gender'].map(str) + self.y['Age'].map(str))
        # print(self.freqFeatures.shape)
        sq = SelectionWrap(selection)
        if(type=="Both"):
            df = sq.fit_transform(self.features, self.y['Gender'].map(str) + self.y['Age'].map(str))
        else:
            df = sq.fit_transform(self.features, self.y[type])
        print(df.shape)
        return df

    def applyExtraction(self, selection):
        print(self.features.shape)
        sq = ExtractionWrap(selection)
        df = sq.fit_transform(self.features)
        print(df.shape)
        return df

    def getFeatures(self, selection, mode):
        if(type(selection) is PCA):
            result = pd.concat([self.X, self.y,
                                self.applyExtraction(selection)],
                               axis=1)
        else:
            result = pd.concat([self.X, self.y,
                                self.applySelection(selection, mode)],
                               axis=1)
        return result

    def useLasso(self, mode):
        lr = LogisticRegression(penalty="l1", dual=False).fit(self.features, self.y)
        model = SelectFromModel(lr, prefit=True)
        X_new = model.transform(self.features)
        print(X_new.get_support)
        result = pd.DataFrame(data=X_new.todense(), columns=X_new.get_support())
        return pd.concat([self.X, self.y,
                                result],
                               axis=1)




