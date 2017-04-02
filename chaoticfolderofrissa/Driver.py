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


def prepareFeatures(X , y):
    posSeqPipeline = Pipeline([
                            ('get_top', POSSeqWrap())
                      ])

    posSeqFeatures = posSeqPipeline.fit_transform(X)
    posSeqFeatures.to_csv('data/posSequence_features.csv')

    frequencyPipeline = Pipeline([
                            ('extract', ItemSelector('Text')),
                            ('count', CountWrap()),
                            ('tfidf', TFIDFWrap())
                        ])

    frequencyFeatures = frequencyPipeline.fit_transform(X)
    frequencyFeatures.to_csv('data/frequency_features.csv')

    timePipeline = Pipeline([
                            ('extract', ItemSelector('PostTime')),
                            ('enrange', PostTimeWrap())
                        ])

    timeFeatures = timePipeline.fit_transform(X)
    timeFeatures.to_csv('data/time_features.csv')

def dimensionReduction():
    feature=Feature()
    gen_data = feature.getFeatures(selection=SelectKBest(chi2, k=1000), mode='Gender')
    gen_data.to_csv('data/features_chi2_gender.csv')
    gen_data = feature.getFeatures(selection=SelectKBest(chi2, k=1000), mode='Age')
    gen_data.to_csv('data/features_chi2_age.csv')
    gen_data = feature.getFeatures(selection=SelectKBest(chi2, k=1000), mode='Both')
    gen_data.to_csv('data/features_chi2_both.csv')

    gen_data = feature.getFeatures(selection=SVD(n_components=1000), mode='Gender')
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
X, y = DOM().getData()
prepareFeatures(X, y)

#2. Dimension Reduction
# feature=Feature()
# gen_data = feature.getFeatures(selection=SelectKBest(chi2, k=1000), mode='Gender')
# gen_data.to_csv('data/features_chi2_gender.csv')
# age_data = feature.getFeatures(selection=PCA(n_components=1000), mode='Age')
# age_data.to_csv('data/features_pca_age.csv')
# both_data = feature.getFeatures(selection=SelectKBest(mutual_info_classif, k=1000), mode='Both')
# both_data.to_csv('data/features_mi_both.csv')

dimensionReduction()

#2. kFold for Parallel
# age_data = pd.read_csv("data/features_chi2_age.csv", index_col=0, encoding='latin1')
# age_model = RootModel(data=age_data, type='Age', modelType=DecisionTreeClassifier)
# train_results, test_results = age_model.evaluateKfold()
# print(train_results)
# print(test_results)
#
# gen_data = pd.read_csv("data/features_mi_gender.csv", index_col=0, encoding='latin1')
# gen_model = RootModel(data=gen_data, type='Gender', modelType=RidgeClassifier)
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
