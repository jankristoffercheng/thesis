from collections import Counter

import nltk


class POSFeature:
    VERB = 'VB'
    ADJECTIVE = 'JJ'

    def __init__(self, text):
        self.nVerbs = 0
        self.nAdjectives = 0
        self.sPOS = ''
        #self.getPOSCount(text)
        self.getPOSTag(text)

    def getPOSCount(self, text):
        tokenizedText = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokenizedText)
        counts = Counter(tag for word, tag in tagged)
        for key, value in counts.items():
            if key.startswith(self.VERB):
                self.nVerbs += value
            if key.startswith(self.ADJECTIVE):
                self.nAdjectives += value

    def getPOSTag(self, text):
        tokenizedText = nltk.word_tokenize(text)
        posTagged = nltk.pos_tag(tokenizedText)
        posTagged = ['-' + tag[1] for tag in posTagged]
        self.sPOS = ''.join(posTagged)[1:]
        self.sPOS = "'"+self.sPOS+"'"
        print("POS:", self.sPOS)


