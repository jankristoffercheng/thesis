from features.TFIDF import TFIDF
from utility.PostCleaner import PostCleaner


class EmojisEmoticons:

    def __init__(self):
        self.tfidf = TFIDF()

    def getEmojiTFIDF(self, data):
        # postDAO = PostsDAO()
        postCleaner = PostCleaner()
        emojiList = []
        for index, post in data.iteritems():
            result = ' '.join(postCleaner.getEmojis(post))

            if result != '':
                #result = result.replace('b\'\\u', '\'\\u')
                emojiList.append(result.encode('unicode_escape'))
            else:
                emojiList.append("")

        print("emoji post:",emojiList)
        return self.tfidf.get_training_TFIDF(emojiList)

    def getLabels(self):
        return self.tfidf.getFeatureNames()

# ee = EmojisEmoticons()
# print(ee.getEmojiTFIDF())