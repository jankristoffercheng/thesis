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

        #get filipino tags from tagalog tagger and map them to nltk tags
        filTags = ['NN', 'NN', 'NN', 'UNK', 'JJ']

        jvmPath = jpype.getDefaultJVMPath()
        jpype.startJVM(jvmPath,"-Djava.class.path=dependencies/RBPOST.jar")
        rbpost = JPackage("rbpost").RBPOST
        hello = rbpost.hPOST_Text(text)

        print('HELLO: ' + hello)

        jpype.shutdownJVM()

        #final tags
        posTags = []

        langDetector = LanguageDetector()
        language = langDetector.getLanguage(text)
        for i in range(len(engTags)):
            if(engTags[i] == filTags):
                posTags.append(engTags[i])
            elif(language == Language.FILIPINO and filTags[i] != self.UNKNOWN):
                posTags.append(filTags[i])
            else:
                posTags.append(engTags[i])

        posTags = ['-' + tag for tag in posTags]
        self.sPOS = ''.join(posTags)[1:]
        self.sPOS = "'"+self.sPOS+"'"
        print(language)
        print(engTags)
        print(filTags)
        print("POS:", self.sPOS)


