from classification.MultinomialNaive import MultinomialNaive
from classification.SVM import SVM
from classification.DecisionTree import DecisionTree

from dao.PostsDAO import PostsDAO
from dao.UsersDAO import UsersDAO
from features.POSFeature import POSFeature
from features.POSSequencePattern import POSSequencePattern


class Controller:
    def __init__(self):
        self.usersDAO = UsersDAO()
        self.postsDAO = PostsDAO()
        self.svm = SVM()
        self.multinomial = MultinomialNaive()
        self.decisionTree = DecisionTree()

    def addUser(self, username, gender, birthyear, birthmonth, birthday, source):
        self.usersDAO.addUser(username, gender, birthyear, birthmonth, birthday, source)

    def addPost(self, username, text, hour, min):
        self.postsDAO.addPost(username, text, hour, min)

    def classifyGenderUsingSVM(self, text):
        trainingData = self.postsDAO.getTrainingData()
        posFeature = POSFeature(text)
        test = [posFeature.nVerbs, posFeature.nAdjectives]


        return self.svm.classify(trainingData['Dimensions'], trainingData['Classes'], test)


    def classifyGenderUsingSVMwithPersist(self, clf,  text):
        posFeature = POSFeature(text)
        test = [posFeature.nVerbs, posFeature.nAdjectives]

        return self.svm.classifyPersist(clf, test)

    def trainSVM(self):
        print('getting')
        trainingData = self.postsDAO.getTrainingData()
        print('training')
        return self.svm.train(trainingData['Dimensions'], trainingData['Classes'])

    def trainGenderPreferential(self):
        print('getting')
        trainingData = self.postsDAO.getTrainingGenderPreferentialData()
        print('training')
        clf = self.multinomial.train(trainingData['Dimensions'], trainingData['Classes'])
        self.multinomial.classifyPersist(clf, ['dota', 'lipstick','butt'])

    def trainPOSSequencePatternWithTree(self):
        print('getting data [POS]')
        trainingData = self.postsDAO.getTrainingPOSData()
        print('traiing data [POS]')
        clf = self.decisionTree.train(trainingData['Dimensions'], trainingData['Classes'])

        posSequencePattern =  POSSequencePattern(trainingData['Documents'])
        print('mining POS Sequence Patterns')
        minedPOSSeqPatterns = posSequencePattern.minePOSPatterns(0.3,0.7)
        print('finish mining POS Sequence Patterns: ', minedPOSSeqPatterns)
        self.decisionTree.classifyPersist(clf, minedPOSSeqPatterns)