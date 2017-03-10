from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from features.TFIDF import TFIDF


#remember to train the tfidf first!
class MultinomialNaive:


    def __init__(self):
        self.tfidf = TFIDF()

    def train(self, dimensions, classes):

        X_train_tfidf = self.tfidf.get_training_TFIDF(dimensions)

        clf = MultinomialNB()
        clf.fit(X_train_tfidf, classes)
        #clf = svm.SVC(kernel='linear')
        #clf = clf.fit(X_train_tfidf, classes)

        return clf

    def classify(self, dimensions, classes, test):
        clf = self.train(dimensions, classes)

        X_new_tfidf = self.tfidf.get_testing_TFIDF(test)

        predicted = clf.predict(X_new_tfidf)

        for doc, category in zip(test, predicted):
            print('%r => %s' % (doc, category))

        return predicted

    def classifyPersist(self, clf, test):
        X_new_tfidf = self.tfidf.get_testing_TFIDF(test)

        predicted = clf.predict(X_new_tfidf)
        for doc, category in zip(test, predicted):
            print('%r => %s' % (doc, category))

        return predicted

