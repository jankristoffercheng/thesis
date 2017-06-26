import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

class Context:
    def process(self,s):
        s = s.lower()
        s = s.translate({ord(c): " " for c in string.punctuation})
        words = s.split()
        # print(words)
        next_words = []
        for x in range(len(words)):
            if words[x] == "my" and x + 1 < len(words):
                next_words.append(words[x + 1])

        return " ".join(next_words)
    # return next_words
#
# s = "my pet is my love. My hope, my salvation. my"
# s2 = "my shoes are black"
# cv = CountVectorizer(tokenizer=tokenize)
# X_train_counts = cv.fit_transform([s,s2])
#
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#
# clf = MultinomialNB().fit(X_train_tfidf, [0,1])
#
# docs_new = ['my dog is my pet', 'just bought my shoes']
# X_new_counts = cv.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#
# predicted = clf.predict(X_new_tfidf)
#
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, category))