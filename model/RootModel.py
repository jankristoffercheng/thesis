import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


class RootModel:
    """
    This class represents the parallel and combined structure. Its results can be fed to the StackModel for the stacked model structure.
    """
    def __init__(self, data, type, modelType, k=10):
        """
        :param data: data to be used for training and testing
        :param type: Gender or Age
        :param modelType: The type of classifier to utilize
        :param k: k-fold crossvalidation
        """
        self.data = data
        self.type=type
        self.models=[]
        self.__kFold(modelType)

    def getTrainingX(self, ind):
        """
        :param ind: k-fold index
        :return: training data for the ith k-fold
        """
        temp = self.data.loc[self.data['Batch'] != ind+1]
        # if(self.type=='Age'):
        #     temp=pd.concat([temp, self.data.loc[self.data['Batch'] != ind+1]['Gender']], axis=1)
        # else:
        #     temp=pd.concat([temp, self.data.loc[self.data['Batch'] != ind+1]['Age']], axis=1)
        # print(temp.iloc[:, :].columns.values, temp.iloc[:, 4:].columns.values)
        return temp.iloc[:, 4:]
    def getTrainingy(self, ind):
        """
        :param ind: k-fold index
        :return: training results for the ith k-fold
        """
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp[self.type]
    def getTrainingUser(self, ind):
        """
        :param ind: k-fold index
        :return: users for the training data for the ith k-fold
        """
        temp = self.data.loc[self.data['Batch'] != ind+1]
        return temp['User']

    def getTestingX(self, ind):
        """
        :param ind: k-fold index
        :return: testing data for the ith k-fold
        """
        temp = self.data.loc[self.data['Batch'] == ind+1]
        # if(self.type=='Age'):
        #     temp=pd.concat([temp, self.data.loc[self.data['Batch'] == ind+1]['Gender']], axis=1)
        # else:
        #     temp=pd.concat([temp, self.data.loc[self.data['Batch'] == ind+1]['Age']], axis=1)

        return temp.iloc[:, 4:]
    def getTestingy(self, ind):
        """
        :param ind: k-fold index
        :return: testing results for the ith k-fold
        """
        temp = self.data.loc[self.data['Batch'] == ind+1]
        return temp[self.type]
    def getTestingUser(self, ind):
        """
        :param ind: k-fold index
        :return: users for the testing data for the ith k-fold
        """
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
        """
        :param train_predictions: predictions of the model for the training data
        :param test_predictions:  predictions of the model for the testing data
        :return: returns the metrics for both training data and testing data
        """
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

            train_precision_results['User'].append(metrics.precision_score(trainY, train_predictions[i], average='macro'))
            test_precision_results['User'].append(metrics.precision_score(testY, test_predictions[i], average='macro'))

            train_recall_results['User'].append(metrics.recall_score(trainY, train_predictions[i], average='macro'))
            test_recall_results['User'].append(metrics.recall_score(testY, test_predictions[i], average='macro'))

            train_kappa_results['User'].append(metrics.cohen_kappa_score(trainY, train_predictions[i]))
            test_kappa_results['User'].append(metrics.cohen_kappa_score(testY, test_predictions[i]))

            train_fmeasure_results['User'].append(metrics.f1_score(trainY, train_predictions[i], average='macro'))
            test_fmeasure_results['User'].append(metrics.f1_score(testY, test_predictions[i], average='macro'))


        return [sum(train_accuracy_results['User'])/len(train_accuracy_results['User']),
                sum(train_precision_results['User'])/len(train_precision_results['User']),
                sum(train_recall_results['User'])/len(train_recall_results['User']),
                sum(train_kappa_results['User'])/len(train_kappa_results['User']),
                sum(train_fmeasure_results['User'])/len(train_fmeasure_results['User'])], [
                sum(test_accuracy_results['User'])/len(test_accuracy_results['User']),
                sum(test_precision_results['User'])/len(test_precision_results['User']),
                sum(test_recall_results['User'])/len(test_recall_results['User']),
                sum(test_kappa_results['User'])/len(test_kappa_results['User']),
                sum(test_fmeasure_results['User'])/len(test_fmeasure_results['User'])]

    def getPredictions(self):
        """
        :return: the predictions of the model for training and testing data
        """
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
        """creates and trains the models to be used for the 10-fold crossvalidation

        :param modelType: the type of model to be created
        """
        self.models = []
        for i in range(0,10):
            # print("train")
            if(modelType is svm.SVC):
                model = modelType(kernel='linear')
            elif(modelType is MultinomialNB):
                model = modelType()
            elif(modelType is RidgeClassifier):
                model = modelType(alpha=1.0)
            elif(modelType is DecisionTreeClassifier):
                model = modelType(criterion='entropy',min_samples_split=20, random_state=99)

            tX, ty = self.getTrainingX(i), self.getTrainingy(i)

            model = model.fit(tX, ty)

            self.models.append(model)
