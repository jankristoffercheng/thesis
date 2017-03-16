from collections import Counter

import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB

from chaoticfolderofrissa.DOM import DOM
from connection.Connection import Connection


##### Get data and split into features and results
# X, y = DOM().getData()

##### Generate files for features
# posSeqPipeline = Pipeline([
#                         ('get_top', POSSeqWrap())
#                   ])
#
# posSeqFeatures = posSeqPipeline.fit_transform(X)
# posSeqFeatures.to_csv('data/posSequence_features.csv')
#
# frequencyPipeline = Pipeline([
#                         ('extract', ItemSelector('Text')),
#                         ('count', CountWrap()),
#                         ('tfidf', TFIDFWrap())
#                     ])
#
# frequencyFeatures = frequencyPipeline.fit_transform(X)
# frequencyFeatures.to_csv('data/frequency_features.csv')
#
# timePipeline = Pipeline([
#                         ('extract', ItemSelector('PostTime')),
#                         ('enrange', PostTimeWrap())
#                     ])
#
# timeFeatures = timePipeline.fit_transform(X)
# timeFeatures.to_csv('data/time_features.csv')


##### Apply feature selection on frequency
# df = pd.read_csv("data/frequency_features.csv", index_col=0, encoding='latin1')
# print(df.shape)
# sq = SelectionWrap(SelectKBest(chi2, k=1000))
# df = sq.fit_transform(df, y['Gender'])
# print(df.shape)
# df.to_csv('data/frequency_features_chi2_gender.csv')


##### Save all features with results into one csv
# result = pd.concat([X,y, pd.read_csv("data/time_features.csv", index_col=0), pd.read_csv("data/posSequence_features.csv", index_col=0), pd.read_csv("data/frequency_features_chi2_gender.csv", index_col=0, encoding='latin1')], axis=1)
# print(result)
# result.to_csv('data/features_chi2_gender.csv')


##### Generate model and view 10-fold results
data = pd.read_csv("data/features_chi2_gender.csv", index_col=0, encoding='latin1')
X=data.iloc[:,6:]
y=data['Gender']

