from sklearn.pipeline import Pipeline

from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.pipelinewraps.CountWrap import CountWrap
from chaoticfolderofrissa.pipelinewraps.ItemSelector import ItemSelector
from chaoticfolderofrissa.pipelinewraps.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.pipelinewraps.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.pipelinewraps.SelectionWrap import SelectionWrap
from chaoticfolderofrissa.pipelinewraps.TFIDFWrap import TFIDFWrap
from connection.Connection import Connection
import pandas as pd


class Feature:
    def __init__(self):
        self.X,self.y = DOM().getData()
        self.timeFeatures = pd.read_csv("data/time_features.csv", index_col=0)
        self.posFeatures = pd.read_csv("data/posSequence_features.csv", index_col=0)
        self.freqFeatures = pd.read_csv("data/frequency_features.csv", index_col=0, encoding='latin1')

    def applySelection(self, selection, type):
        print(self.freqFeatures.shape)
        sq = SelectionWrap(selection)
        df = sq.fit_transform(self.freqFeatures, self.y[type])
        print(df.shape)
        return df

    def getFeatures(self, selection, type):
        result = pd.concat([self.X, self.y, self.timeFeatures,
                            self.posFeatures,
                            self.applySelection(selection, type)],
                           axis=1)
        return result


