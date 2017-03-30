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
from chaoticfolderofrissa.pipelinewraps.CountWrap import CountWrap
from chaoticfolderofrissa.pipelinewraps.ItemSelector import ItemSelector
from chaoticfolderofrissa.pipelinewraps.LIWCWrap import LIWCWrap
from chaoticfolderofrissa.pipelinewraps.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.pipelinewraps.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.pipelinewraps.TFIDFWrap import TFIDFWrap


##### Generate files for features
from features.Selection import Selection
from features.TFIDF import TFIDF


def prepareFeatures(X , y):
    # posSeqPipeline = Pipeline([
    #                         ('get_top', POSSeqWrap())
    #                   ])

    posSeqFeatures = pd.read_csv("data/posSequence_features.csv", index_col=0)

    timePipeline = Pipeline([
                            ('extract', ItemSelector('PostTime')),
                            ('enrange', PostTimeWrap())
                        ])

    timeFeatures = timePipeline.fit_transform(X)

    liwcFeatures = pd.read_csv("data/liwc_features.csv", index_col=0)

    tfidf = TFIDF()
    freq = tfidf.get_training_TFIDF(X['Text'])
    for type in ['Gender', 'Age', 'Both']:
        if(type!='Both'):
            target = y[type]
        else:
            target = y['Gender'].map(str) + y['Age'].map(str)

        sel, col = Selection().featureSelectChi2(1000, freq, target)
        result = pd.DataFrame(data=sel.todense(), columns=np.asarray(tfidf.getFeatureNames())[col])
        results = pd.concat([X, y, timeFeatures,  posSeqFeatures, result, liwcFeatures], axis=1)
        results.to_csv('data/features_chi2_'+type+'.csv')

        sel = Selection().featureExtractSVD(1000, freq, target)
        result = pd.DataFrame(data=sel)
        results = pd.concat([X, y, timeFeatures, posSeqFeatures, result, liwcFeatures], axis=1)
        results.to_csv('data/features_svd_' + type + '.csv')

        sel, col = Selection().featureSelectMutualClassif(1000, freq, target)
        result = pd.DataFrame(data=sel.todense(), columns=np.asarray(tfidf.getFeatureNames())[col])
        results = pd.concat([X, y, timeFeatures, posSeqFeatures, result, liwcFeatures], axis=1)
        results.to_csv('data/features_mi_' + type + '.csv')

def dimensionReduction():
    feature=Feature()
    gen_data = feature.getFeatures(selection=SelectKBest(chi2, k=1000), mode='Gender')
    gen_data.to_csv('data/features_chi2_gender.csv')
    gen_data = feature.getFeatures(selection=SelectKBest(chi2, k=1000), mode='Age')
    gen_data.to_csv('data/features_chi2_age.csv')
    gen_data = feature.getFeatures(selection=SelectKBest(chi2, k=1000), mode='Both')
    gen_data.to_csv('data/features_chi2_both.csv')

    gen_data = feature.getFeatures(selection=PCA(n_components=1000), mode='Gender')
    gen_data.to_csv('data/features_pca_gender.csv')
    gen_data = feature.getFeatures(selection=PCA(n_components=1000), mode='Age')
    gen_data.to_csv('data/features_pca_age.csv')
    gen_data = feature.getFeatures(selection=PCA(n_components=1000), mode='Both')
    gen_data.to_csv('data/features_pca_both.csv')

    gen_data = feature.getFeatures(selection=SelectKBest(mutual_info_classif, k=1000), mode='Gender')
    gen_data.to_csv('data/features_mi_gender.csv')
    gen_data = feature.getFeatures(selection=SelectKBest(mutual_info_classif, k=1000), mode='Age')
    gen_data.to_csv('data/features_mi_age.csv')
    gen_data = feature.getFeatures(selection=SelectKBest(mutual_info_classif, k=1000), mode='Both')
    gen_data.to_csv('data/features_mi_both.csv')

#1. Prepare features
# X, y = DOM().getData()
# prepareFeatures(X, y)

#2. Dimension Reduction
# dimensionReduction()

#2. kFold for Parallel
# age_data = pd.read_csv("data/features_chi2_age.csv", index_col=0, encoding='latin1')
# age_model = RootModel(data=age_data, type='Age', modelType=MultinomialNB)
# train_results, test_results = age_model.evaluateKfold()
# print(train_results)
# print(test_results)
#
# gen_data = pd.read_csv("data/features_mi_Gender.csv", index_col=0, encoding='latin1')
# gen_model = RootModel(data=gen_data, type='Gender', modelType=Ridge)
# train_results, test_results = gen_model.evaluateKfold()
# print(train_results)
# print(test_results)

#3. Stack
# gen_data = pd.read_csv("data/features_mi_gender.csv", index_col=0, encoding='latin1')
# gen_model = StackModel(root=gen_model, data=gen_data, type='Gender', modelType=svm.SVC)
# train_results, test_results = gen_model.evaluateKfold()
# print(train_results)
# print(test_results)
#
# age_data = pd.read_csv("data/features_pca_age.csv", index_col=0, encoding='latin1')
# age_model = StackModel(root=gen_model, data=gen_data, type='Age', modelType=MultinomialNB)
# train_results, test_results = age_model.evaluateKfold()
# print(train_results)
# print(test_results)

#4. Combined
# both_data = pd.read_csv("data/features_chi2_both.csv", index_col=0, encoding='latin1')
# both_model = RootModel(data=both_data, type='Gender', modelType=MultinomialNB)
# train_results, test_results = both_model.evaluateKfold()
# print(train_results)
# print(test_results)
#
# both_data = pd.read_csv("data/features_chi2_both.csv", index_col=0, encoding='latin1')
# both_model = RootModel(data=both_data, type='Age', modelType=MultinomialNB)
# train_results, test_results = both_model.evaluateKfold()
# print(train_results)
# print(test_results)
