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

    def getPOSTag(self, post):

        #assumptions: engPOS and filPOS are strings that contains POS tags sperated by -
        tokenizedText = nltk.word_tokenize(post.content)

        #step 1: split tweets/posts into sentences
        text = ' '.join(tokenizedText)
        sentences = []
        finalPOSTags = ""
        posTags =[]
        punctuations = ['.', '?', '!']
        prevIndex = 0
        for i in range(len(text)):
            if text[i] in punctuations:
                sentences.append(text[prevIndex:i+1].strip())
                prevIndex = i+1

        engTags = [tag for tag in post.epos.split("-")]

        posDAO = PosDAO()
        posMapDict = posDAO.getPOSMapping()

        filTags = [posMapDict.get(tag, tag) for tag in post.fpos.split("-")]


        startIndex = 0

        #step 2: detect the language of each sentence
        langDetector = LanguageDetector()
        for sentence in sentences:

            language = langDetector.getLanguage(sentence)
            if(language == Language.FILIPINO):
                for startIndex in range(len(filTags)):
                    finalPOSTags = finalPOSTags + "-" + filTags[startIndex]

            elif(language == Language.ENGLISH):
                for startIndex in range(len(engTags)):
                    finalPOSTags = finalPOSTags + "-" + engTags[startIndex]

            else:
                for startIndex in range(len(engTags)):
                    if(engTags[startIndex] == filTags[startIndex]):
                        finalPOSTags = finalPOSTags + "-" + engTags[startIndex]
                    elif (filTags[i][:2] == engTags[i][:2]):
                        finalPOSTags = finalPOSTags + "-" + filTags[startIndex][:2]
                    else:
                        finalPOSTags = finalPOSTags + "-" + engTags[startIndex]

            startIndex = startIndex + len(sentence) - 1

        print(engTags)
        print(filTags)
        print("POS:", finalPOSTags)

