from collections import Counter
import nltk

class POSSequencePattern:

    MAX_LENGTH = 7
    def getPOSTags(self, text):
        tokenizedText = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokenizedText)
        return tagged

    def getPOSFreq(self, countPOS, nLen, minsup):
        freqResult = []

        for key, value in countPOS.items():
            print("Value/nLen: ", value/nLen, " >= 0.30")
            if value/nLen >= minsup:
                freqResult[key] = value

        print("feq: ",freqResult)
        return freqResult



    def getPOSCount(self, tweet):
        tokenizedText = nltk.word_tokenize(tweet)
        tagged = nltk.pos_tag(tokenizedText)
        print(nltk.pos_tag(tokenizedText))
        counts = Counter(tag for word, tag in tagged)
        return counts

        # for key, value in counts.items():
        #     print("key: ", key, " value: ", value)
            # if key.startswith(self.VERB):
            #     self.nVerbs += value
            # if key.startswith(self.ADJECTIVE):
            #     self.nAdjectives += value

    def minePOSPatterns(self, document, numDocument, tagSet, minsup, minadherence):
        pLen = 1 #pattern length
        countPOS = self.getPOSCount(text)
        frequencyPOS = self.getPOSFreq(countPOS, numDocument, minsup)

        seqPattern = frequencyPOS[:1]

text = "I am Shayane Tan and she is Rissa Quindoza."
p = POSSequencePattern()
p.minePOSPatterns(text, p.getPOSTags(text), 0.30, 0)