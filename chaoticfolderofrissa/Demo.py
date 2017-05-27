from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.Feature import Feature
from chaoticfolderofrissa.RootModel import RootModel
from chaoticfolderofrissa.StackModel import StackModel
from chaoticfolderofrissa.pipelinewraps.AgeRangeWrap import AgeRangeWrap
from chaoticfolderofrissa.pipelinewraps.CountWrap import CountWrap
from chaoticfolderofrissa.pipelinewraps.GenderWrap import GenderWrap
from chaoticfolderofrissa.pipelinewraps.ItemSelector import ItemSelector
from chaoticfolderofrissa.pipelinewraps.LIWCWrap import LIWCWrap
from chaoticfolderofrissa.pipelinewraps.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.pipelinewraps.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.pipelinewraps.TFIDFWrap import TFIDFWrap


##### Generate files for features
from features.Selection import Selection
from features.TFIDF import TFIDF


def prepareFeatures(X , y):
    y['Gender'] = GenderWrap().fit_transform(y['Gender'])
    y['Age'] = AgeRangeWrap().fit_transform(y['Age'])

    posSeqPipeline = Pipeline([
                            ('get_top', POSSeqWrap())
                      ])

    posSeqFeatures = posSeqPipeline.fit_transform(X)
    # posSeqFeatures.to_csv("data/posSequence_features.csv")

    timePipeline = Pipeline([
                            ('extract', ItemSelector('PostTime')),
                            ('enrange', PostTimeWrap())
                        ])

    timeFeatures = timePipeline.fit_transform(X)
    # timeFeatures.to_csv("data/time_features.csv")

    liwcFeatures = pd.read_csv("data/liwc_features.csv", index_col=0)
    liwcFeatures.columns = ["LIWC."+str(col) for col in liwcFeatures.columns.values]
    # liwcFeatures.to_csv("data/liwc_features.csv")

    tfidf = TFIDF()
    freq = tfidf.get_training_TFIDF(X['Text'])

    sel, col = Selection().featureSelectChi2(1000, freq, y['Gender'])
    result = pd.DataFrame(data=sel.todense(), columns=np.asarray(tfidf.getFeatureNames())[col])
    results = pd.concat([X, y, timeFeatures, posSeqFeatures, result, liwcFeatures], axis=1)
    results.to_csv('data2/features_chi2_' + 'Gender' + '.csv')

    sel = Selection().featureExtractSVD(1000, freq, y['Age'])
    result = pd.DataFrame(data=sel, columns=["Frq." + str(num) for num in range(1000)])
    results = pd.concat([X, y, timeFeatures, posSeqFeatures, result, liwcFeatures], axis=1)
    results.to_csv('data2/features_svd_' + 'Age' + '.csv')


def getSpecificFeatures(data, features):
    filter_col = [col for col in list(data) if (("." not in col) or (col.startswith(tuple(features))))]
    return data[filter_col]


#1. Prepare features
X, y = DOM().getData()
prepareFeatures(X, y)

#2. Dimension Reduction
# dimensionReduction()

#2. kFold for Parallel

# gen_data = pd.read_csv("data2/features_chi2_Gender.csv", index_col=0, encoding='latin1')
# gen_model = RootModel(data=gen_data, type='Gender', modelType=DecisionTreeClassifier)
# train_results, test_results = gen_model.evaluateKfold()
# print(train_results)
# print(test_results)
#
# age_data = pd.read_csv("data2/features_svd_Age.csv", index_col=0, encoding='latin1')
# age_model = RootModel(data=age_data, type='Age', modelType=RidgeClassifier)
# train_results, test_results = age_model.evaluateKfold()
# print(train_results)
# print(test_results)

