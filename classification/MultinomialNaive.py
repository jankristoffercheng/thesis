
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

class MultinomialNaive:


    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def train(self, dimensions, classes):
        dtm  = self.vectorizer.fit_transform(dimensions)
        vocab = self.vectorizer.get_feature_names()

        X_train_tfidf = self.tfidf_transformer.fit_transform(dtm)

        # clf = MultinomialNB().fit(X_train_tfidf, classes)
        clf = svm.SVC(kernel='linear')
        clf = clf.fit(X_train_tfidf, classes)
        return clf

    def classify(self, dimensions, classes, test):
        clf = self.train(dimensions, classes)

        X_new_counts = self.vectorizer.transform(test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)

        predicted = clf.predict(X_new_tfidf)

        for doc, category in zip(test, predicted):
            print('%r => %s' % (doc, category))

        return predicted

    def classifyPersist(self, clf, test):
        X_new_counts = self.vectorizer.transform(test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)

        predicted = clf.predict(X_new_tfidf)
        for doc, category in zip(test, predicted):
            print('%r => %s' % (doc, category))

        return predicted

