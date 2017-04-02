from collections import Counter

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


class RootModel:

    def __init__(self, data, type, modelType, k=10):
        self.user = data['User']
        self.X = data.iloc[:, 6:]
        self.y = data[type]
        self.type=type

        self.train_index=[]
        self.test_index=[]
        for train, test in GroupKFold(n_splits = k).split(self.X, self.y, groups=self.user):
            self.train_index.append(train)
            self.test_index.append(test)
        self.models=[]
        self.__kFold(modelType)

    def __evaluateUserFold(self, predictions, user, X, y):
        useres = []
        trueres = []
        ind = 0
        i = 1
        while (i < len(X.index)):
            if (user.loc[i] != user.loc[ind] or i == len(X.index) - 1):
                if(i==len(X.index) -1):
                    df = predictions[ind:i]
                else:
                    df = predictions[ind:i - 1]

                counter = Counter(df)
                useres.append(counter.most_common(1)[0][0])
                trueres.append(y.loc[ind])
                ind = i

            i += 1

        return useres, trueres

    def evaluateKfold(self, train_predictions=None, test_predictions=None):
        if(train_predictions is None or test_predictions is None):
            train_predictions, test_predictions = self.getPredictions()

        train_results = {'Post':[],'User':[]}
        test_results = {'Post':[],'User':[]}

        for ind, train_index, test_index in zip(range(len(self.train_index)), self.train_index, self.test_index):
            # print("evaluate")
            trainUser = self.user.iloc[train_index].reset_index(drop=True)
            trainX = self.X.iloc[train_index].reset_index(drop=True)
            trainY = self.y.iloc[train_index].reset_index(drop=True)

            useres, trueres = self.__evaluateUserFold(train_predictions[ind], trainUser, trainX, trainY)
            train_results['Post'].append(metrics.accuracy_score(trainY, pd.Series(train_predictions[ind])))
            train_results['User'].append(metrics.accuracy_score(trueres, useres))

            testUser = self.user.iloc[test_index].reset_index(drop=True)
            testX = self.X.iloc[test_index].reset_index(drop=True)
            testY = self.y.iloc[test_index].reset_index(drop=True)

            useres, trueres = self.__evaluateUserFold(test_predictions[ind], testUser, testX, testY)
            test_results['Post'].append(metrics.accuracy_score(testY, pd.Series(test_predictions[ind])))
            test_results['User'].append(metrics.accuracy_score(trueres, useres))

        return train_results, test_results

    def getPredictions(self):
        train_predictions = []
        test_predictions = []
        for ind, train_index, test_index in zip(range(len(self.train_index)), self.train_index, self.test_index):
            trainX = self.X.iloc[train_index].reset_index(drop=True)
            train_pred = self.models[ind].predict(trainX)
            train_predictions.append(train_pred.tolist())

            testX = self.X.iloc[test_index].reset_index(drop=True)
            test_pred = self.models[ind].predict(testX)
            test_predictions.append(test_pred.tolist())

        return train_predictions,test_predictions



    def __kFold(self, modelType):
        self.models = []
        for train_index, test_index in zip(self.train_index, self.test_index):
            # print("train")
            if(modelType is svm.SVC):
                model = modelType(kernel='linear')
            elif(modelType is MultinomialNB):
                model = modelType()
            elif(modelType is RidgeClassifier):
                model = modelType(alpha=1.0)
            elif(modelType is DecisionTreeClassifier):
                model = modelType(criterion='entropy',min_samples_split=20, random_state=99)

            model.fit(self.X.iloc[train_index], self.y.iloc[train_index])

            self.models.append(model)
