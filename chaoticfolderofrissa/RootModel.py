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
        # self.user = data['User']
        self.data = data
        # self.X = data.iloc[:, 6:]
        # self.y = data[type]
        self.type=type
        #
        # self.train_index=[]
        # self.test_index=[]
        # for train, test in GroupKFold(n_splits = k).split(self.X, self.y, groups=self.user):
        #     self.train_index.append(train)
        #     self.test_index.append(test)
        self.models=[]
        self.__kFold(modelType)

    def getTrainingX(self, ind):
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp.iloc[:, 7:]
    def getTrainingy(self, ind):
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp[type]
    def getTrainingUser(self, ind):
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp['User']

    def getTestingX(self, ind):
        temp = self.data.loc[self.data['Batch'] == ind+1]
        return temp.iloc[:, 7:]
    def getTestingy(self, ind):
        temp = self.data.loc[self.data['Batch'] == ind+1]
        return temp[type]
    def getTestingUser(self, ind):
        temp = self.data.loc[self.data['Batch'] == ind+1]
        return temp['User']
    #
    # def __evaluateUserFold(self, predictions, user, X, y):
    #     useres = []
    #     trueres = []
    #     ind = 0
    #     i = 1
    #     while (i < len(X.index)):
    #         if (user.loc[i] != user.loc[ind] or i == len(X.index) - 1):
    #             if(i==len(X.index) -1):
    #                 df = predictions[ind:i]
    #             else:
    #                 df = predictions[ind:i - 1]
    #
    #             counter = Counter(df)
    #             useres.append(counter.most_common(1)[0][0])
    #             trueres.append(y.loc[ind])
    #             ind = i
    #
    #         i += 1
    #
    #     return useres, trueres

    def evaluateKfold(self, train_predictions=None, test_predictions=None):
        if(train_predictions is None or test_predictions is None):
            train_predictions, test_predictions = self.getPredictions()

        train_results = {'Post':[],'User':[]}
        test_results = {'Post':[],'User':[]}

        for i in range(0,10):
            # print("evaluate")
            # trainUser = self.getTrainingUser(i).reset_index(drop=True)
            # trainX = self.getTrainingX(i).reset_index(drop=True)
            trainY = self.getTrainingy(i).reset_index(drop=True)

            # useres, trueres = self.__evaluateUserFold(train_predictions[i], trainUser, trainX, trainY)
            train_results['User'].append(metrics.accuracy_score(trainY, pd.Series(train_predictions[i])))
            # train_results['User'].append(metrics.accuracy_score(trueres, useres))

            # testUser = self.getTestingUser(i).reset_index(drop=True)
            # testX = self.getTestingX(i).reset_index(drop=True)
            testY = self.getTestingy(i).reset_index(drop=True)

            # useres, trueres = self.__evaluateUserFold(test_predictions[i], testUser, testX, testY)
            # print(test_predictions[i])
            # print(useres)
            test_results['User'].append(metrics.accuracy_score(testY, pd.Series(test_predictions[i])))
            # test_results['User'].append(metrics.accuracy_score(trueres, useres))

        return train_results, test_results

    def getPredictions(self):
        train_predictions = []
        test_predictions = []
        for i in range(0,10):
            trainX = self.getTrainingX(i).reset_index(drop=True)
            train_pred = self.models[i].predict(trainX)
            train_predictions.append(train_pred.tolist())

            testX = self.getTestingX(i).reset_index(drop=True)
            test_pred = self.models[i].predict(testX)
            test_predictions.append(test_pred.tolist())

        return train_predictions,test_predictions



    def __kFold(self, modelType):
        self.models = []
        for i in range(0,10):
            print("train")
            if(modelType is svm.SVC):
                model = modelType()
            elif(modelType is MultinomialNB):
                model = modelType()
            elif(modelType is RidgeClassifier):
                model = modelType(alpha=1.0)
            elif(modelType is DecisionTreeClassifier):
                model = modelType(criterion='entropy',min_samples_split=20, random_state=99)

            tX, ty = self.getTrainingX(i), self.getTrainingy(i)

            model.fit(tX, ty)

            self.models.append(model)
