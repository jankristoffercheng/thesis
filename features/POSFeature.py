from collections import Counter

import nltk

from utility.LanguageDetector import LanguageDetector, Language
import jpype
from jpype import *

class POSFeature:
    VERB = 'VB'
    ADJECTIVE = 'JJ'
    UNKNOWN = 'UNK'

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
        engTags = [tag[1] for tag in posTagged]
        #final tags
        posTags = engTags

        '''
        langDetector = LanguageDetector()
        language = langDetector.getLanguage(text)
        for i in range(len(engTags)):
            if(engTags[i] == filTags):
                posTags.append(engTags[i])
            elif(language == Language.FILIPINO and filTags[i] != self.UNKNOWN):
                posTags.append(filTags[i])
            else:
                posTags.append(engTags[i])
        '''

        posTags = ['-' + tag for tag in posTags]
        self.sPOS = ''.join(posTags)[1:]

