from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.Feature import Feature
from chaoticfolderofrissa.FeatureExtract import FeatureExtract
from chaoticfolderofrissa.RootModel import RootModel
from chaoticfolderofrissa.StackModel import StackModel
from chaoticfolderofrissa.pipelinewraps.AgeRangeWrap import AgeRangeWrap
from chaoticfolderofrissa.pipelinewraps.CharacterWrap import CharacterWrap
from chaoticfolderofrissa.pipelinewraps.ContextualWrap import ContextualWrap
from chaoticfolderofrissa.pipelinewraps.CountWrap import CountWrap
from chaoticfolderofrissa.pipelinewraps.DisfluencyWrap import DisfluencyWrap
from chaoticfolderofrissa.pipelinewraps.EmojiWrap import EmojiWrap
from chaoticfolderofrissa.pipelinewraps.FunctionWrap import FunctionWrap
from chaoticfolderofrissa.pipelinewraps.GenderWrap import GenderWrap
from chaoticfolderofrissa.pipelinewraps.ItemSelector import ItemSelector
from chaoticfolderofrissa.pipelinewraps.LIWCWrap import LIWCWrap
from chaoticfolderofrissa.pipelinewraps.LinkWrap import LinkWrap
from chaoticfolderofrissa.pipelinewraps.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.pipelinewraps.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.pipelinewraps.StructureWrap import StructureWrap
from chaoticfolderofrissa.pipelinewraps.TFIDFWrap import TFIDFWrap
from chaoticfolderofrissa.pipelinewraps.WordWrap import WordWrap
from features.Selection import Selection
from features.TFIDF import TFIDF
from utility.DataCleaner import DataCleaner


def clean(x):
    return  DataCleaner().clean_data(x)

def dimensionReduction(X ,y, source, data=None):
    feature=Feature(X, y, source, data)
    gen_data = feature.getFeatures(selection=SelectFpr(chi2), mode='Gender')
    gen_data.to_csv('data/'+source+'/features_chi2_gender.csv')
    gen_data = feature.getFeatures(selection=SelectFpr(chi2), mode='Age')
    gen_data.to_csv('data/'+source+'/features_chi2_age.csv')
    gen_data = feature.getFeatures(selection=SelectFpr(chi2), mode='Both')
    gen_data.to_csv('data/'+source+'/features_chi2_both.csv')


    gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=100), mode='Gender')
    gen_data.to_csv('data/'+source+'/features_svd_gender.csv')
    gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=100), mode='Age')
    gen_data.to_csv('data/'+source+'/features_svd_age.csv')
    gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=100), mode='Both')
    gen_data.to_csv('data/'+source+'/features_svd_both.csv')

    gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=20), mode='Gender')
    gen_data.to_csv('data/'+source+'/features_mi_gender.csv')
    gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=20), mode='Age')
    gen_data.to_csv('data/'+source+'/features_mi_age.csv')
    gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=20), mode='Both')
    gen_data.to_csv('data/'+source+'/features_mi_both.csv')


    gen_data = feature.useLasso(mode='Gender')
    gen_data.to_csv('data/'+source+'/features_lasso_gender.csv')
    gen_data = feature.useLasso(mode='Age')
    gen_data.to_csv('data/'+source+'/features_lasso_age.csv')
    gen_data = feature.useLasso(mode='Both')
    gen_data.to_csv('data/'+source+'/features_lasso_both.csv')

def getSpecificFeatures(data, features):
    filter_col = [col for col in list(data) if (("." not in col) or (col.startswith(tuple(features))))]
    return data[filter_col]

def evaluate(file, source):
    age_data = pd.read_csv("data/"+source+"/"+file+"_age.csv", index_col=0, encoding='latin1')
    gen_data = pd.read_csv("data/"+source+"/"+file+"_gender.csv", index_col=0, encoding='latin1')
    both_data = pd.read_csv("data/"+source+"/"+file+"_both.csv", index_col=0, encoding='latin1')


    age_model = RootModel(data=age_data, type='Age', modelType=svm.SVC)
    train_results, test_results = age_model.evaluateKfold()
    print(train_results)
    print(test_results)

    gen_model = StackModel(root=age_model, data=gen_data, type='Gender', modelType=svm.SVC)
    train_results, test_results = gen_model.evaluateKfold()
    print(train_results)
    print(test_results)

    gen_model = RootModel(data=gen_data, type='Gender', modelType=svm.SVC)
    train_results, test_results = gen_model.evaluateKfold()
    print(train_results)
    print(test_results)

    age_model = StackModel(root=gen_model, data=gen_data, type='Age', modelType=svm.SVC)
    train_results, test_results = age_model.evaluateKfold()
    print(train_results)
    print(test_results)

    both_model = RootModel(data=both_data, type='Gender', modelType=svm.SVC)
    train_results, test_results = both_model.evaluateKfold()
    print(train_results)
    print(test_results)

    both_model = RootModel(data=both_data, type='Age', modelType=svm.SVC)
    train_results, test_results = both_model.evaluateKfold()
    print(train_results)
    # print(test_results)



X, y = DOM().getTwitterData()
UX, Uy = DOM().getTwitterUserData()
source="twitter"

#1. Prepare features
# fe = FeatureExtract("twitter")
# data = pd.concat([X, fe.get_liwc(), fe.fit_transform(X)],axis=1)
# data = data.iloc[:,7:].groupby(data['User']).mean()
# maxmin = MinMaxScaler()
# data=pd.DataFrame(data=maxmin.fit_transform(data), columns=data.columns)
# data.to_csv('data/'+source+'/raw/features_fin.csv')

#3. Dimension Reduction
# dimensionReduction(UX, Uy, "twitter")

#2. kFold for Parallel
evaluate("features_chi2","twitter")
