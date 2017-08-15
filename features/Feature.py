from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from pipelinewraps.AgeRangeWrap import AgeRangeWrap
from pipelinewraps.ExtractionWrap import ExtractionWrap
from pipelinewraps.GenderWrap import GenderWrap
from pipelinewraps.SelectionWrap import SelectionWrap
import pandas as pd
import numpy as np

class Feature:
    """
    Applies dimension reduction to data
    """
    def __init__(self, X, y, source, data=None):
        """
        :param X: text data
        :param y: classes (gender and age)
        :param source: twitter, facebook, or merged
        :param data: features of the data
        """
        self.X, self.y = X, y
        self.y['Gender'] = GenderWrap().fit_transform(self.y['Gender'])
        self.y['Age'] = AgeRangeWrap().fit_transform(self.y['Age'])
        if data is None:
            self.features = pd.read_csv("data/"+source+"/raw/features_fin.csv", encoding = "ISO-8859-1", index_col=0)
        else:
            self.features = data

    def applySelection(self, selection, type):
        """
        applies feature selection

        :param selection: feature selection technique
        :param type: Gender, Age, or Both
        :return: feature selected data
        """
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
        """
        applies feature selection

        :param selection: feature extraction technique
        :param type: Gender, Age, or Both
        :return: feature extracted data
        """
        print(self.features.shape)
        sq = ExtractionWrap(selection)
        df = sq.fit_transform(self.features)
        print(df.shape)
        return df

    def getFeatures(self, selection, mode):
        """
        applies feature selection or extraction

        :param selection: feature selection or extraction technique
        :param mode: Gender, Age, or Both
        :return: feature selected or extracted data
        """
        if(type(selection) is TruncatedSVD):
            result = pd.concat([self.X, self.y,
                                self.applyExtraction(selection)],
                               axis=1)
        else:
            result = pd.concat([self.X, self.y,
                                self.applySelection(selection, mode)],
                               axis=1)
        return result

    def useLasso(self, mode):
        """
        applies LASSO feature selection

        :param selection: feature selection or extraction technique
        :return: feature selected data
        """

        if (mode == "Both"):
            lr = LogisticRegression(penalty="l1", dual=False).fit(self.features, self.y['Gender'].map(str) + self.y['Age'].map(str))
        else:
            lr = LogisticRegression(penalty="l1", dual=False).fit(self.features, self.y[mode])
        model = SelectFromModel(lr, prefit=True)
        X_new = model.transform(self.features)
        orgcol = self.features.columns
        cols=[]
        print(orgcol)
        print(model.get_support())
        for stat, labl in zip(model.get_support(), orgcol):
            print(stat,type(stat))
            if(stat==True): cols.append(labl)

        print(X_new.shape)
        print(cols)
        result = pd.DataFrame(data=X_new, columns=cols)
        return pd.concat([self.X, self.y,
                                result],
                               axis=1)




