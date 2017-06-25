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

    for i in range(10, 61, 10):
        gen_data = feature.getFeatures(selection=SelectPercentile(chi2, percentile=i), mode='Gender')
        gen_data.to_csv('data/'+source+'/chi2/gender_'+i+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(chi2, percentile=i), mode='Age')
        gen_data.to_csv('data/'+source+'/chi2/age_'+i+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(chi2, percentile=i), mode='Both')
        gen_data.to_csv('data/'+source+'/chi2/both_'+i+'.csv')

    for i in range(100,1001,100):
        gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=i), mode='Gender')
        gen_data.to_csv('data/'+source+'/svd/gender_'+i+'.csv')
        gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=i), mode='Age')
        gen_data.to_csv('data/'+source+'/svd/age_'+i+'.csv')
        gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=i), mode='Both')
        gen_data.to_csv('data/'+source+'/svd/both_'+i+'.csv')

    for i in range(10,61,10):
        gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=i), mode='Gender')
        gen_data.to_csv('data/'+source+'/mi/gender_'+i+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=i), mode='Age')
        gen_data.to_csv('data/'+source+'/mi/age_'+i+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=i), mode='Both')
        gen_data.to_csv('data/'+source+'/mi/both_'+i+'.csv')

    gen_data = feature.useLasso(mode='Gender')
    gen_data.to_csv('data/'+source+'/lasso/gender.csv')
    gen_data = feature.useLasso(mode='Age')
    gen_data.to_csv('data/'+source+'/lasso/age.csv')
    gen_data = feature.useLasso(mode='Both')
    gen_data.to_csv('data/'+source+'/lasso/both.csv')

def getSpecificFeatures(data, features):
    filter_col = [col for col in list(data) if (("." not in col) or (col.startswith(tuple(features))))]
    return data[filter_col]

def get_Data_from_CSV(source, fs, param=None):
    if(fs=="lasso"):
        age_data = pd.read_csv("data/" + source + "/" + fs + "/age_"+param+".csv", index_col=0, encoding='latin1')
        gen_data = pd.read_csv("data/" + source + "/" + fs + "/gender_"+param+".csv", index_col=0, encoding='latin1')
        both_data = pd.read_csv("data/" + source + "/" + fs + "/both_"+param+".csv", index_col=0, encoding='latin1')
    else:
        age_data = pd.read_csv("data/" + source + "/" + fs + "/age_"+param+".csv", index_col=0, encoding='latin1')
        gen_data = pd.read_csv("data/" + source + "/" + fs + "/gender_"+param+".csv", index_col=0, encoding='latin1')
        both_data = pd.read_csv("data/" + source + "/" + fs + "/both_"+param+".csv", index_col=0, encoding='latin1')

    return age_data,gen_data,both_data

def evaluate(age_data, gen_data, both_data, model):
    age_model = RootModel(data=age_data, type='Age', modelType=model)
    train_results, test_results = age_model.evaluateKfold()
    print("Parallel Model, Age, Train: " + train_results)
    print("Root Model, Age, Results: " + test_results)

    gen_model = StackModel(root=age_model, data=gen_data, type='Gender', modelType=model)
    train_results, test_results = gen_model.evaluateKfold()
    print("Stacked Model, Gender, Train: " + train_results)
    print("Stacked Model, Gender, Results: " + test_results)

    gen_model = RootModel(data=gen_data, type='Gender', modelType=model)
    train_results, test_results = gen_model.evaluateKfold()
    print("Parallel Model, Gender, Train: " + train_results)
    print("Parallel Model, Gender, Results: " + test_results)

    age_model = StackModel(root=gen_model, data=gen_data, type='Age', modelType=model)
    train_results, test_results = age_model.evaluateKfold()
    print("Stacked Model, Age, Train: " + train_results)
    print("Stacked Model, Age, Results: " + test_results)

    both_model = RootModel(data=both_data, type='Gender', modelType=model)
    train_results, test_results = both_model.evaluateKfold()
    print("Combined Model, Gender, Train: " + train_results)
    print("Combined Model, Gender, Results: " + test_results)

    both_model = RootModel(data=both_data, type='Age', modelType=model)
    train_results, test_results = both_model.evaluateKfold()
    print("Combined Model, Age, Train: " + train_results)
    print("Combined Model, Age, Results: " + test_results)



X, y = DOM().getTwitterData()
UX, Uy = DOM().getTwitterUserData()
source="twitter"


CLASSIFIERS = {
	"SVM": svm.SVC,
	"NB": MultinomialNB,
    "Ridge": RidgeClassifier,
    "DTC": DecisionTreeClassifier
}
SOURCES = [
    "twitter",
    "facebook",
    "merged"
]
FEATURE_REDUCTIONS = [
	["lasso", None],
    ["svd", 100],
    ["svd", 200],
    ["svd", 300],
    ["svd", 400],
    ["svd", 500],
    ["svd", 600],
    ["svd", 700],
    ["svd", 800],
    ["svd", 900],
    ["svd", 1000],
    ["mi", 10],
    ["mi", 20],
    ["mi", 30],
    ["mi", 40],
    ["mi", 50],
    ["mi", 60],
    ["chi2", 10],
    ["chi2", 20],
    ["chi2", 30],
    ["chi2", 40],
    ["chi2", 50],
    ["chi2", 60],
]

#1. Prepare features
fe = FeatureExtract("twitter")
data = pd.concat([X, fe.get_liwc(), fe.fit_transform(X)],axis=1)
data = data.iloc[:,7:].groupby(data['User']).mean()
maxmin = MinMaxScaler()
data=pd.DataFrame(data=maxmin.fit_transform(data), columns=data.columns)
data.to_csv('data/'+source+'/raw/features_fin.csv')

#3. Dimension Reduction
# dimensionReduction(UX, Uy, "twitter")

#2. kFold for Parallel

# for source in SOURCES:
#     for fr in FEATURE_REDUCTIONS:
#         for classifier in CLASSIFIERS:
#             age_data, gen_data, both_data = get_Data_from_CSV(source, fr[0], fr[1])
#             evaluate(age_data, gen_data, both_data, classifier)
#             print(source + " " + fr + " " + classifier + ": ")
