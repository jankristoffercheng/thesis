from sklearn.decomposition import PCA
from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.pipelinewraps.AgeRangeWrap import AgeRangeWrap
from chaoticfolderofrissa.pipelinewraps.ExtractionWrap import ExtractionWrap
from chaoticfolderofrissa.pipelinewraps.GenderWrap import GenderWrap
from chaoticfolderofrissa.pipelinewraps.SelectionWrap import SelectionWrap
import pandas as pd


class Feature:

    def __init__(self):
        self.X,self.y = DOM().getData()
        self.y['Gender'] = GenderWrap().fit_transform(self.y['Gender'])
        self.y['Age'] = AgeRangeWrap().fit_transform(self.y['Age'])
        self.timeFeatures = pd.read_csv("data/time_features.csv", index_col=0)
        self.posFeatures = pd.read_csv("data/posSequence_features.csv", index_col=0)
        self.freqFeatures = pd.read_csv("data/frequency_features.csv", index_col=0, encoding='latin1')
        self.liwcFeatures = pd.read_csv("data/liwc_features.csv", index_col=0)

    def applySelection(self, selection, type):
        # print(self.y[['Gender', 'Age']])
        # print(self.y['Gender'].map(str) + self.y['Age'].map(str))
        # print(self.freqFeatures.shape)
        sq = SelectionWrap(selection)
        if(type=="Both"):
            df = sq.fit_transform(self.freqFeatures, self.y['Gender'].map(str) + self.y['Age'].map(str))
        else:
            df = sq.fit_transform(self.freqFeatures, self.y[type])
        print(df.shape)
        return df

    def applyExtraction(self, selection):
        print(self.freqFeatures.shape)
        sq = ExtractionWrap(selection)
        df = sq.fit_transform(self.freqFeatures)
        print(df.shape)
        return df

    def getFeatures(self, selection, mode):
        if(type(selection) is PCA):
            result = pd.concat([self.X, self.y, self.timeFeatures,
                                self.posFeatures,
                                self.applyExtraction(selection),
                                self.liwcFeatures],
                               axis=1)
        else:
            result = pd.concat([self.X, self.y, self.timeFeatures,
                                self.posFeatures,
                                self.applySelection(selection, mode),
                                self.liwcFeatures],
                               axis=1)
        return result


