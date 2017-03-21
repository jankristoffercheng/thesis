from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from chaoticfolderofrissa.DOM import DOM
from chaoticfolderofrissa.Feature import Feature
from chaoticfolderofrissa.RootModel import RootModel
from chaoticfolderofrissa.StackModel import StackModel
from chaoticfolderofrissa.pipelinewraps.CountWrap import CountWrap
from chaoticfolderofrissa.pipelinewraps.ItemSelector import ItemSelector
from chaoticfolderofrissa.pipelinewraps.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.pipelinewraps.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.pipelinewraps.TFIDFWrap import TFIDFWrap


def prepareFeatures(X , y):


    ##### Generate files for features
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


#1. Prepare features
# X, y = DOM().getData()
# prepareFeatures(X, y)

#2. kFold for Parallel
gen_data = Feature().getFeatures(selection=SelectKBest(chi2, k=1000), type='Gender')
gen_model = RootModel(data=gen_data, type='Gender', modelType=MultinomialNB)
train_results, test_results = gen_model.evaluateKfold()
print(train_results)
print(test_results)

age_data = Feature().getFeatures(selection=SelectKBest(chi2, k=1000), type='Age')
age_model = StackModel(root=gen_model, data=gen_data, type='Age', modelType=MultinomialNB)
train_results, test_results = age_model.evaluateKfold()
print(train_results)
print(test_results)


