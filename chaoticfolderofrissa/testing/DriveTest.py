import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.Feature import Feature
from chaoticfolderofrissa.FeatureExtract import FeatureExtract
from utility.DataCleaner import DataCleaner


def clean(x):
    return  DataCleaner().clean_data(x)

def averaging(data):
    return data.groupby(['User'])[:,7:].mean()

fe = FeatureExtract()
X, y = DOM().getTwitterData()
UX, Uy = DOM().getTwitterUserData()
source="twitter"
train_predictions = []
test_predictions = []




for i in range(0,10):
    trainingX = X.loc[X['Batch']!=i]
    trainingy = y.loc[y['Batch']!=i]
    testingX = X.loc[X['Batch']==i]
    testingy = y.loc[y['Batch']==i]

    data = pd.concat([UX, averaging(fe.fit_transform(trainingX))], axis=1)

    types = ['Gender', 'Age', 'Both']
    feature = Feature(X, y, source, data)
    for type in types:
        gen_data = feature.getFeatures(selection=SelectFpr(chi2), mode=type)
        gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=1000), mode=type)
        gen_data = feature.getFeatures(selection=SelectFpr(mutual_info_classif), mode=type)

    testdata = fe.transform(testingX)
