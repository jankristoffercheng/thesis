from collections import Counter

import nltk

from dao.PosDAO import PosDAO
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

        posDAO = PosDAO()
        posMapDict = posDAO.getPOSMapping()

        jvmPath = jpype.getDefaultJVMPath()
        jpype.startJVM(jvmPath,"-Djava.class.path=dependencies/RBPOST.jar")
        rbpost = JPackage("rbpost").RBPOST

        hposTags = rbpost.hPOST_Text(' '.join(tokenizedText)).split()

        filTags = [posMapDict.get(tag, tag) for tag in hposTags]
        jpype.shutdownJVM()

        #final tags
        posTags = engTags


        langDetector = LanguageDetector()

        for i in range(len(engTags)):
            language = langDetector.getLanguage(posTagged[i][0])
            print("text:",posTagged[i][0]," language: ",language)
            if(engTags[i] == filTags[i]):
                posTags.append(engTags[i])
            elif(language == Language.FILIPINO):
                if(filTags[i] != self.UNKNOWN):
                    posTags.append(filTags[i])
                elif(filTags[i][:2] == engTags[i][:2]):
                    posTags.append(filTags[i][:2])

            else:
                posTags.append(engTags[i])


        posTags = ['-' + tag for tag in posTags]
        self.sPOS = ''.join(posTags)[1:]
        print(engTags)
        print(filTags)
        print("POS:", self.sPOS)

