from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class TFIDF:
    """
    Processes the TFIDF of text
    """
    def __init__(self, mindf, maxdf):
        """
        :param mindf: lower threshold for term frequency filter
        :param maxdf: upper threshold for term frequency filter
        """
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer(stop_words='english', max_df=maxdf, min_df=mindf)

    def getFeatureNames(self):
        """
        :return: labels of the features
        """
        return ["Frq." + freq for freq in self.vectorizer.get_feature_names()]

    def get_training_TFIDF(self, documentList):
        """
        :param documentList: training text data
        :return: tfidf of the text
        """
        dtm  = self.vectorizer.fit_transform(documentList)
        X_train_tfidf = self.tfidf_transformer.fit_transform(dtm)

        return  X_train_tfidf

    def get_testing_TFIDF(self, test):
        """
        :param documentList: testing text data
        :return: tfidf of the text
        """
        X_new_counts = self.vectorizer.transform(test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)

        return  X_new_tfidf


