from collections import Counter

from sklearn import metrics, svm
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from chaoticfolderofrissa.pipelinewraps.StackAgeRangeWrap import StackAgeRangeWrap
from chaoticfolderofrissa.pipelinewraps.StackGenderWrap import StackGenderWrap


class StackModel:
    def __init__(self, root, modelType, data, type, k=10):
        self.root=root
        self.data= data
        self.dataList=[]
        for i in range(0,10):
            self.dataList.append(data)
        self.models=[]
        self.type=type
        self.__kFold(modelType)

    def getTrainingX(self, ind):
        temp = self.dataList[ind].loc[self.dataList[ind]['Batch'] != ind+1]
        return temp.iloc[:, 4:]
    def getTrainingy(self, ind):
        temp = self.dataList[ind].loc[self.dataList[ind]['Batch'] != ind+1]
        return temp[self.type]
    def getTrainingUser(self, ind):
        temp = self.dataList[ind].loc[self.dataList[ind]['Batch'] != ind+1]
        return temp['User']
    def getTestingX(self, ind):
        temp = self.dataList[ind].loc[self.dataList[ind]['Batch'] == ind+1]
        return temp.iloc[:, 4:]
    def getTestingy(self, ind):
        temp = self.dataList[ind].loc[self.dataList[ind]['Batch'] == ind+1]
        return temp[self.type]
    def getTestingUser(self, ind):
        temp = self.dataList[ind].loc[self.dataList[ind]['Batch'] == ind+1]
        return temp['User']

    def __evaluateUserFold(self, predictions, user, X, y):
        useres = []
        trueres = []
        ind = 0
        i = 1
        while (i < len(X.index)):
            if (user.loc[i] != user.loc[ind] or i == len(X.index) - 1):
                if (i == len(X.index) - 1):
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

        train_results = {'Post': [], 'User': []}
        test_results = {'Post': [], 'User': []}

        for i in range(0,10):
            trainY = self.getTrainingy(i).reset_index(drop=True)
            train_results['User'].append(metrics.accuracy_score(trainY, train_predictions[i]))
            testY = self.getTestingy(i).reset_index(drop=True)
            test_results['User'].append(metrics.accuracy_score(testY, test_predictions[i]))

        return train_results, test_results

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

        return train_predictions, test_predictions

    def __kFold(self, modelType):
        self.models = []
        root_training, root_testing = self.root.getPredictions()
        for i in root_training:
            print(len(i))
        for i in range(0,10):
            if (self.root.type == "Age"):
                agewrap = StackAgeRangeWrap()
                root_train = agewrap.transform(root_training[i])
                root_test = agewrap.transform(root_testing[i])
            else:
                genwrap = StackGenderWrap()
                root_train = genwrap.transform(root_training[i])
                root_test = genwrap.transform(root_testing[i])

            pd.concat([root_train, root_test], axis=0).to_csv("concat.csv")

            self.dataList[i] = pd.concat([self.dataList[i], pd.concat([root_train,root_test],axis=0)], axis=1)
            # print(self.dataList[i].shape)

            if (modelType is svm.SVC):
                model = modelType()
            elif (modelType is MultinomialNB):
                model = modelType()
            elif (modelType is RidgeClassifier):
                model = modelType(alpha=1.0)
            elif (modelType is DecisionTreeClassifier):
                model = modelType(criterion='entropy', min_samples_split=20, random_state=99)

            self.getTrainingX(i).to_csv("trainingx.csv")
            self.getTrainingy(i).to_csv("trainingy.csv")
            print(i, self.getTrainingX(i).shape, self.getTrainingy(i).shape)
            model.fit(self.getTrainingX(i), self.getTrainingy(i))

            self.models.append(model)
