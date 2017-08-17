from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from features.TFIDF import TFIDF
from utility.PostCleaner import PostCleaner


class EmojisEmoticons:


    def __init__(self):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer(stop_words='english')

    def getEmojiTFIDF(self, data):
        """
        :param data: text to be processed
        :return: TFIDF of the extracted emojis
        """
        postCleaner = PostCleaner()
        emojiList = []
        for index, post in data.iteritems():
            result = ' '.join(postCleaner.getEmojis(post))

            if result != '':
                emojiList.append(result.encode('unicode_escape'))
            else:
                emojiList.append("")

        dtm = self.vectorizer.fit_transform(emojiList)
        return self.tfidf_transformer.fit_transform(dtm)

    def getLabels(self):
        return self.vectorizer.get_feature_names()
