from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class TFIDF:
    def __init__(self):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer()

    def getFeatureNames(self):
        return ["Frq." + freq for freq in self.vectorizer.get_feature_names()]

    def get_training_TFIDF(self, documentList):
        dtm  = self.vectorizer.fit_transform(documentList)
        X_train_tfidf = self.tfidf_transformer.fit_transform(dtm)

        return  X_train_tfidf

    def get_testing_TFIDF(self, test):
        X_new_counts = self.vectorizer.transform(test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)

        return  X_new_tfidf


