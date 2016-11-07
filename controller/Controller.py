from classification.SVM import SVM
from dao.PostsDAO import PostsDAO
from dao.UsersDAO import UsersDAO
from features.POSFeature import POSFeature


class Controller:
    def __init__(self):
        self.usersDAO = UsersDAO()
        self.postsDAO = PostsDAO()
        self.svm = SVM()

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