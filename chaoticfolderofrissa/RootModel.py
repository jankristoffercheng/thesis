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
import numpy as np

class RootModel:

    def __init__(self, data, type, modelType, k=10):
        self.data = data
        self.type=type
        self.models=[]
        self.__kFold(modelType)

    def getTrainingX(self, ind):
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp.iloc[:, 4:]
    def getTrainingy(self, ind):
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp[self.type]
    def getTrainingUser(self, ind):
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp['User']

    def getTestingX(self, ind):
        temp = self.data.loc[self.data['Batch'] == ind+1]
        return temp.iloc[:, 4:]
    def getTestingy(self, ind):
        temp = self.data.loc[self.data['Batch'] == ind+1]
        return temp[self.type]
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

        train_accuracy_results = {'Post':[],'User':[]}
        test_accuracy_results = {'Post':[],'User':[]}

        train_precision_results = {'Post': [], 'User': []}
        test_precision_results = {'Post': [], 'User': []}

        train_recall_results = {'Post': [], 'User': []}
        test_recall_results = {'Post': [], 'User': []}

        train_kappa_results = {'Post': [], 'User': []}
        test_kappa_results = {'Post': [], 'User': []}

        train_fmeasure_results = {'Post': [], 'User': []}
        test_fmeasure_results = {'Post': [], 'User': []}

        for i in range(0,10):
            trainY = self.getTrainingy(i)
            testY = self.getTestingy(i)

            train_accuracy_results['User'].append(metrics.accuracy_score(trainY, train_predictions[i]))
            test_accuracy_results['User'].append(metrics.accuracy_score(testY, test_predictions[i]))

            train_precision_results['User'].append(metrics.precision_score(trainY, train_predictions[i], average='micro'))
            test_precision_results['User'].append(metrics.precision_score(testY, test_predictions[i], average='micro'))

            train_recall_results['User'].append(metrics.recall_score(trainY, train_predictions[i], average='micro'))
            test_recall_results['User'].append(metrics.recall_score(testY, test_predictions[i], average='micro'))

            train_kappa_results['User'].append(metrics.cohen_kappa_score(trainY, train_predictions[i]))
            test_kappa_results['User'].append(metrics.cohen_kappa_score(testY, test_predictions[i]))

            train_fmeasure_results['User'].append(metrics.f1_score(trainY, train_predictions[i], average='micro'))
            test_fmeasure_results['User'].append(metrics.f1_score(testY, test_predictions[i], average='micro'))

        return [train_accuracy_results, train_precision_results, train_recall_results, train_kappa_results, train_fmeasure_results], [test_accuracy_results, test_precision_results, test_recall_results,  test_kappa_results, test_fmeasure_results]

    def getPredictions(self):
        train_predictions = []
        test_predictions = []
        for i in range(0,10):
            trainX = self.getTrainingX(i)
            train_pred = self.models[i].predict(trainX)
            train_predictions.append(pd.Series(data=train_pred, index=trainX.index.values))

            testX = self.getTestingX(i)
            test_pred = self.models[i].predict(testX)
            test_predictions.append(pd.Series(data=test_pred, index=testX.index.values))

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

            model = model.fit(tX, ty)

            self.models.append(model)
