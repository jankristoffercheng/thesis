import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def train(self, dimensions, classes):
        dtm = self.vectorizer.fit_transform(dimensions)
        vocab = self.vectorizer.get_feature_names()

        X_train_tfidf = self.tfidf_transformer.fit_transform(dtm)

        clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
        clf.fit(X_train_tfidf, classes)
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
