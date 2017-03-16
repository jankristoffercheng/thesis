from collections import Counter
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from chaoticfolderofrissa.CountWrap import CountWrap
from chaoticfolderofrissa.ItemSelector import ItemSelector
from chaoticfolderofrissa.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.SelectionWrap import SelectionWrap
from chaoticfolderofrissa.TFIDFWrap import TFIDFWrap
from chaoticfolderofrissa.VectorizeWrap import VectorizeWrap
from connection.Connection import Connection


def getData():
    conn = Connection().getConnection()
    cursor = conn.cursor()

    sql = "SELECT P.User, P.Text, hour(P.PostTime) as Time, P.POS, (DATE_FORMAT(CURDATE(), '%Y') - DATE_FORMAT(U.Birthdate, '%Y') - (DATE_FORMAT(CURDATE(), '00-%m-%d') < DATE_FORMAT(U.Birthdate, '00-%m-%d'))) AS Age, U.Gender  FROM post P, user U WHERE P.User = U.Id"
    cursor.execute(sql)
    rows = cursor.fetchall()
    data = {'Features': [[row['User'],row['Text'],row['Time'], row['POS']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}

    return data

##### Get data and split into features and results
# data = getData()
# X=pd.DataFrame(data['Features'], columns=['User','Text', 'PostTime', 'POS'])
# y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

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

gkf = GroupKFold(n_splits=10)
for train_index, test_index in gkf.split(X, y, groups=data['User']):
    model = MultinomialNB()
    model.fit(X.iloc[train_index], data['Gender'].iloc[train_index])

    s = joblib.dump(model, "cm/parallel/mnb_chi2_gender.pkl")

    trainData = data.iloc[train_index].reset_index(drop=True)
    trainX = X.iloc[train_index].reset_index(drop=True)
    trainY = y.iloc[train_index].reset_index(drop=True)
    testData = data.iloc[test_index].reset_index(drop=True)
    testX = X.iloc[test_index].reset_index(drop=True)
    testY = y.iloc[test_index].reset_index(drop=True)

    print("Posts Confusion Matrix & Accuracy")

    train_results = pd.Series(model.predict(trainX))
    print('cm', confusion_matrix(trainY, train_results))
    print('acc', metrics.accuracy_score(trainY, train_results))

    test_results = pd.Series(model.predict(testX))
    print('cm', confusion_matrix(testY, test_results))
    print('acc', metrics.accuracy_score(testY, test_results))

    print("Users Confusion Matrix & Accuracy")

    #### Training data
    useres = []
    trueres = []
    ind = 0
    i = 1
    while (i < len(trainX.index)):
        if (trainData.loc[i]['User'] != trainData.loc[ind]['User']):
            df = trainX.loc[ind:i - 1]
            pred = model.predict(df)
            counter = Counter(pred)
            useres.append(counter.most_common(1)[0][0])
            trueres.append(trainY.loc[ind])
            ind = i

        i += 1

    print('cm', confusion_matrix(trueres, useres))
    print('acc', metrics.accuracy_score(trueres, useres))

    useres = []
    trueres = []
    ind = 0
    i = 1
    while (i < len(testX.index)):
        if (testData.loc[i]['User'] != testData.loc[ind]['User']):
            df = testX.loc[ind:i - 1]
            pred = model.predict(df)
            counter = Counter(pred)
            useres.append(counter.most_common(1)[0][0])
            trueres.append(testY.loc[ind])
            ind = i

        i += 1

    print('cm', confusion_matrix(trueres, useres))
    print('acc', metrics.accuracy_score(trueres, useres))