from dao.PostsDAO import PostsDAO
from features.TFIDF import TFIDF
from utility.PostCleaner import PostCleaner


class EmojisEmoticons:

    def getEmojiTFIDF(self):
        postDAO = PostsDAO()
        postCleaner = PostCleaner()
        emojiList = []
        for post in postDAO.getAllPost():
            result = ' '.join(postCleaner.getEmojis(post))

            if result != '':
                #result = result.replace('b\'\\u', '\'\\u')
                emojiList.append(result.encode('unicode_escape'))

        print("emoji post:",emojiList)
        tfidf = TFIDF()
        return tfidf.get_training_TFIDF(emojiList)


ee = EmojisEmoticons()
print(ee.getEmojiTFIDF())