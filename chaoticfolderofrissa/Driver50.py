from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
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


##### Generate files for features
from chaoticfolderofrissa.pipelinewraps.WordWrap import WordWrap
from features.Selection import Selection
from features.TFIDF import TFIDF
from utility.DataCleaner import DataCleaner



# def prepareUnitedRawFeatures(X, y, source):

    # results.to_csv("data/" + source + "/raw/features.csv")

def prepareRawFeatures(X , y, source):
    y['Gender'] = GenderWrap().fit_transform(y['Gender'])
    y['Age'] = AgeRangeWrap().fit_transform(y['Age'])


    # tfidf = TFIDF()
    # data = X['Text'].apply(clean)
    # freq = tfidf.get_training_TFIDF(data)
    # for type in ['Age', 'Gender', 'Both']:
    #     if (type != 'Both'):
    #         target = y[type]
    #     else:
    #         target = y['Gender'].map(str) + y['Age'].map(str)
    #
    #     sel, col = Selection().featureSelectChi2(1000, freq, target)
    #     result = pd.DataFrame(data=sel.todense(), columns=np.asarray(tfidf.getFeatureNames())[col])
    #     results = pd.concat(
    #         [X, y, timeFeatures, result, characterFeatures, wordFeatures, structureFeatures, linkFeatures,
    #          socLinFeatures], axis=1)
    #     results.to_csv('data/features_chi2_' + type + '.csv')
    #
    #     sel = Selection().featureExtractSVD(1000, freq, target)
    #     result = pd.DataFrame(data=sel, columns=["Frq." + str(num) for num in range(1000)])
    #     results = pd.concat(
    #         [X, y, timeFeatures, result, characterFeatures, wordFeatures, structureFeatures, linkFeatures,
    #          socLinFeatures], axis=1)
    #     results.to_csv('data/features_svd_' + type + '.csv')
    #
    #     sel, col = Selection().featureSelectMutualClassif(1000, freq, target)
    #     result = pd.DataFrame(data=sel.todense(), columns=np.asarray(tfidf.getFeatureNames())[col])
    #     results = pd.concat(
    #         [X, y, timeFeatures, result, characterFeatures, wordFeatures, structureFeatures, linkFeatures,
    #          socLinFeatures], axis=1)
    #     results.to_csv('data/features_mi_' + type + '.csv')


def clean(x):
    return  DataCleaner().clean_data(x)

def dimensionReduction(X ,y, source, data=None):
    feature=Feature(X, y, source, data)
    # gen_data = feature.getFeatures(selection=SelectFpr(chi2), mode='Gender')
    # gen_data.to_csv('data/'+source+'/features_chi2_gender.csv')
    # gen_data = feature.getFeatures(selection=SelectFpr(chi2), mode='Age')
    # gen_data.to_csv('data/'+source+'/features_chi2_age.csv')
    # gen_data = feature.getFeatures(selection=SelectFpr(chi2), mode='Both')
    # gen_data.to_csv('data/'+source+'/features_chi2_both.csv')

    gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=100), mode='Gender')
    gen_data.to_csv('data/'+source+'/features_svd_gender.csv')
    gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=100), mode='Age')
    gen_data.to_csv('data/'+source+'/features_svd_age.csv')
    gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=100), mode='Both')
    gen_data.to_csv('data/'+source+'/features_svd_both.csv')

    gen_data = feature.getFeatures(selection=SelectFpr(mutual_info_classif), mode='Gender')
    gen_data.to_csv('data/'+source+'/features_mi_gender.csv')
    gen_data = feature.getFeatures(selection=SelectFpr(mutual_info_classif), mode='Age')
    gen_data.to_csv('data/'+source+'/features_mi_age.csv')
    gen_data = feature.getFeatures(selection=SelectFpr(mutual_info_classif), mode='Both')
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

def evaluate(file):
    # age_data = pd.read_csv(file+"Age.csv", index_col=0, encoding='latin1')
    # gen_data = pd.read_csv(file+"Gender.csv", index_col=0, encoding='latin1')
    both_data = pd.read_csv(file+"Both.csv", index_col=0, encoding='latin1')


    # age_model = RootModel(data=age_data, type='Age', modelType=svm.SVC)
    # train_results, test_results = age_model.evaluateKfold()
    # print(train_results)
    # print(test_results)
    #
    # gen_model = StackModel(root=age_model, data=gen_data, type='Gender', modelType=svm.SVC)
    # train_results, test_results = gen_model.evaluateKfold()
    # print(train_results)
    # print(test_results)
    #
    # gen_model = RootModel(data=gen_data, type='Gender', modelType=svm.SVC)
    # train_results, test_results = gen_model.evaluateKfold()
    # print(train_results)
    # print(test_results)
    #
    # age_model = StackModel(root=gen_model, data=gen_data, type='Age', modelType=svm.SVC)
    # train_results, test_results = age_model.evaluateKfold()
    # print(train_results)
    # print(test_results)

    # both_model = RootModel(data=both_data, type='Gender', modelType=svm.SVC)
    # train_results, test_results = both_model.evaluateKfold()
    # print(train_results)
    # print(test_results)

    both_model = RootModel(data=both_data, type='Age', modelType=svm.SVC)
    train_results, test_results = both_model.evaluateKfold()
    print(train_results)
    print(test_results)



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
dimensionReduction(UX, Uy, "twitter")



# data = X[['Text']]
# data.to_csv("merged.csv")
# X, y = DOM().getFBData()
# data = X[['Text']]
# data.to_csv("fb.csv")
# X, y = DOM().getTwitterData()
# data = X[['Text']]
# data.to_csv("twitter.csv")
# cmb = X[['Text', 'Batch']]
# for i in range(1,11):
#         temp = cmb.loc[cmb['Batch'] == i]
#         del temp['Batch']
#         name="datacmb_"+str(i)+".csv"
#         temp.to_csv(name)
# prepareUnitedRawFeatures(X, y, "twitter")
# X, y = DOM().getFBData()
# prepareRawFeatures(X, y, "fb")
# X, y = DOM().getMergedData()
# prepareRawFeatures(X, y, "merge")

#2. kFold for Parallel

# evaluate("data/features_mi_")

# age_data = pd.read_csv("data/features_chi2_age.csv", index_col=0, encoding='latin1')
# age_model = RootModel(data=age_data, type='Age', modelType=MultinomialNB)
# train_results, test_results = age_model.evaluateKfold()
# print(train_results)
# print(test_results)
#
#
# gen_data = pd.read_csv("data/features_mi_Gender.csv", index_col=0, encoding='latin1')
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
