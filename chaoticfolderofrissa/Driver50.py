from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.Feature import Feature
from chaoticfolderofrissa.FeatureExtract import FeatureExtract
from chaoticfolderofrissa.RootModel import RootModel
from chaoticfolderofrissa.StackModel import StackModel
from utility.DataCleaner import DataCleaner

import xlwt


CLASSIFIER_NAMES = ['SVC', 'MultinomialNB', 'RidgeClassifier', 'DecisionTree']
CLASSIFIERS = [
	svm.SVC,
	MultinomialNB,
    RidgeClassifier,
    DecisionTreeClassifier
]
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
DOC_FREQS = [ # min = 1%, 5%, 10%; max = 90%, 80%, 70%
	[0.05, 0.90],
	[0.01, 0.70],
	[0.01, 0.80],
	[0.01, 0.90],
	[0.10, 0.70],
	[0.10, 0.80],
	[0.10, 0.90],
	[0.05, 0.70],
	[0.05, 0.80],
    [0.1, 0.99],
    [0.05, 0.99],
    [0.01, 0.99]
]



def clean(x):
    return  DataCleaner().clean_data(x)

def dimensionReduction(X ,y, source, mindf, maxdf, data=None):
    feature=Feature(X, y, source, data)
    # for freq in DOC_FREQS:
    mindf = str(mindf)
    maxdf = str(maxdf)
    for i in range(10, 61, 10):
        gen_data = feature.getFeatures(selection=SelectPercentile(chi2, percentile=i), mode='Gender')
        gen_data.to_csv('data/'+source+'/chi2/gender_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(chi2, percentile=i), mode='Age')
        gen_data.to_csv('data/'+source+'/chi2/age_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(chi2, percentile=i), mode='Both')
        gen_data.to_csv('data/'+source+'/chi2/both_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')

    for i in range(100,1001,100):
        gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=i), mode='Gender')
        gen_data.to_csv('data/'+source+'/svd/gender_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')
        gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=i), mode='Age')
        gen_data.to_csv('data/'+source+'/svd/age_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')
        gen_data = feature.getFeatures(selection=TruncatedSVD(n_components=i), mode='Both')
        gen_data.to_csv('data/'+source+'/svd/both_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')

    for i in range(10,61,10):
        gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=i), mode='Gender')
        gen_data.to_csv('data/'+source+'/mi/gender_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=i), mode='Age')
        gen_data.to_csv('data/'+source+'/mi/age_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')
        gen_data = feature.getFeatures(selection=SelectPercentile(score_func=mutual_info_classif, percentile=i), mode='Both')
        gen_data.to_csv('data/'+source+'/mi/both_'+str(i)+'_'+mindf+'-'+maxdf+'.csv')

    gen_data = feature.useLasso(mode='Gender')
    gen_data.to_csv('data/'+source+'/lasso/gender_'+mindf+'-'+maxdf+'.csv')
    gen_data = feature.useLasso(mode='Age')
    gen_data.to_csv('data/'+source+'/lasso/age_'+mindf+'-'+maxdf+'.csv')
    gen_data = feature.useLasso(mode='Both')
    gen_data.to_csv('data/'+source+'/lasso/both_'+mindf+'-'+maxdf+'.csv')



def getSpecificFeatures(data, features):
    filter_col = [col for col in list(data) if (("." not in col) or (col.startswith(tuple(features))))]
    return data[filter_col]

def get_Data_from_CSV(source, mindf, maxdf, fs, param=None):
    param = str(param)
    mindf = str(mindf)
    maxdf = str(maxdf)

    if(fs=="lasso"):
        age_data = pd.read_csv("data/" + source + "/" + fs + "/age_"+mindf+"-"+maxdf+".csv", index_col=0, encoding='latin1')
        gen_data = pd.read_csv("data/" + source + "/" + fs + "/gender_"+mindf+"-"+maxdf+".csv", index_col=0, encoding='latin1')
        both_data = pd.read_csv("data/" + source + "/" + fs + "/both_"+mindf+"-"+maxdf+".csv", index_col=0, encoding='latin1')
    else:
        age_data = pd.read_csv("data/" + source + "/" + fs + "/age_"+param+"_"+mindf+"-"+maxdf+".csv", index_col=0, encoding='latin1')
        gen_data = pd.read_csv("data/" + source + "/" + fs + "/gender_"+param+"_"+mindf+"-"+maxdf+".csv", index_col=0, encoding='latin1')
        both_data = pd.read_csv("data/" + source + "/" + fs + "/both_"+param+"_"+mindf+"-"+maxdf+".csv", index_col=0, encoding='latin1')

    return age_data,gen_data,both_data

def evaluate(age_data, gen_data, both_data, model):

    result_collection = {}

    age_model = RootModel(data=age_data, type='Age', modelType=model)
    train_results, test_results = age_model.evaluateKfold()

    result_collection['Parallel_Age_Test'] = test_results

    gen_model = StackModel(root=age_model, data=gen_data, type='Gender', modelType=model)
    train_results, test_results = gen_model.evaluateKfold()

    result_collection['Stacked_Gender_Test'] = test_results

    gen_model = RootModel(data=gen_data, type='Gender', modelType=model)
    train_results, test_results = gen_model.evaluateKfold()

    result_collection['Parallel_Gender_Test'] = test_results

    age_model = StackModel(root=gen_model, data=gen_data, type='Age', modelType=model)
    train_results, test_results = age_model.evaluateKfold()

    result_collection['Stacked_Age_Test'] = test_results

    both_model = RootModel(data=both_data, type='Gender', modelType=model)
    train_results, test_results = both_model.evaluateKfold()

    result_collection['Combined_Gender_Test'] = test_results

    both_model = RootModel(data=both_data, type='Age', modelType=model)
    train_results, test_results = both_model.evaluateKfold()

    result_collection['Combined_Age_Test'] = test_results

    return result_collection

def writeToExcel(book, sheet, classifier, features, row):
    sheet.write(row, 0, classifier)
    for featureName, docFreqs in features.items():
        sheet.write(row, 1, featureName)
        for docFreqVal, models in docFreqs.items():
            sheet.write(row, 2, docFreqVal)
            for modelName, modelMetrics in models.items():
                sheet.write(row, 3, modelName)
                for i in range(len(modelMetrics)):
                    sheet.write(row, 4+i, modelMetrics[i])
                row += 1
    return row

X, y = DOM().getMergedData()
source="facebook"


#1. Prepare features


# for freq in DOC_FREQS:
#     fe = FeatureExtract(source, freq[0], freq[1])
#     data = pd.concat([X, fe.get_liwc(), fe.fit_transform(X)],axis=1)
#     data = data.iloc[:,7:].groupby(data['User']).mean()
#     maxmin = MinMaxScaler()
#     data.to_csv('data/'+source+'/raw/features_init_'+str(freq[0])+'-'+str(freq[1])+'.csv')
#     data=pd.DataFrame(data=maxmin.fit_transform(data), columns=data.columns)
#     data.to_csv('data/'+source+'/raw/features_fin_'+str(freq[0])+'-'+str(freq[1])+'.csv')

#3. Dimension Reduction
#dimensionReduction(UX, Uy, source, 0.01, 0.7)

#for freq in DOC_FREQS:
#    UX, Uy = DOM().getMergedUsersData()
#    features = pd.read_csv("data/"+source+"/raw/features_fin_"+str(freq[0])+"-"+str(freq[1])+".csv", encoding = "ISO-8859-1", index_col=False)
#    features = features.drop(features.columns[0], axis=1)

#    dimensionReduction(UX, Uy, source, freq[0], freq[1], features)


#2. kFold for Parallel

#Metrics order should be according to how the metrics are returned from evaluateKFold method
SHEET_COLUMNS = ["Algorithm", "FReduction_Thrsh","Doc_Freq","Models", "Accuracy", "Precision", "Recall", "Kappa", "F-Measure"]
class_results = {}
book = xlwt.Workbook(encoding="utf-8")
sheet = book.add_sheet("Results",cell_overwrite_ok=True)
for i in range(len(SHEET_COLUMNS)):
    sheet.write(0, i, SHEET_COLUMNS[i])

i = 0
row = 1
for classifier in CLASSIFIERS:
    feature_results = {}
    for fr in FEATURE_REDUCTIONS:
        doc_freq_results = {}
        for freq in DOC_FREQS:
            if (classifier == MultinomialNB and fr[0] != 'svd' or classifier != MultinomialNB):
                age_data, gen_data, both_data = get_Data_from_CSV(source, freq[0], freq[1], fr[0], fr[1])
                doc_freq_results[str(freq[0]) + "_" + str(freq[1])] = evaluate(age_data, gen_data, both_data,
                                                                               classifier)
        feature_results[str(fr[0]) + "_" + str(fr[1])] = doc_freq_results

    row = writeToExcel(book, sheet, CLASSIFIER_NAMES[i], feature_results, row)
    i += 1
book.save('data/'+source+"/"+source+".xls")
#
# X[['Text']].to_csv("twitterdata.csv")


# X, y = DOM().getFBData()
# UX, Uy = DOM().getFBUserData()
# source="fb"
#
# X[['Text']].to_csv("fbdata.csv")


# pd.concat([pd.read_csv('data/fb/raw/features_fin.csv', index_col=0),pd.read_csv('data/twitter/raw/features_fin.csv', index_col=0)],axis=0).to_csv('data/merged/raw/features_fin.csv')

# for freq in DOC_FREQS:
#     pd.concat([pd.read_csv('data/fb/raw/features_fin_'+str(freq[0])+'-'+str(freq[1])+'.csv', index_col=0, encoding = "ISO-8859-1"),pd.read_csv('data/twitter/raw/features_fin_'+str(freq[0])+'-'+str(freq[1])+'.csv', encoding = "ISO-8859-1", index_col=0)],axis=0).to_csv('data/merged/raw/features_fin_'+str(freq[0])+'-'+str(freq[1])+'.csv')