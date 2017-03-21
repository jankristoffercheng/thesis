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
        self.sText = text
        self.sPOS = ''
        self.POSDao = PosDAO()
        self.mapping = {}

        #self.populateMappingDictionary()

        #self.getPOSCount(text)
        #self.getPOSTag(text)

    def populateMappingDictionary(self):
        print("populating map dictionary...")
        with open("mapping.txt", "r") as file_object:
            for line in file_object:
                splitline = line.split()
                self.mapping[' '.join(splitline[1:])] = splitline[0]

    def getEnglishPOS(self):
        tokenizedText = nltk.word_tokenize(self.sText)
        pos = nltk.pos_tag(tokenizedText)
        pos = '-'.join([posTag[1] for posTag in pos])
        return pos

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

        print("beginning to combine pos...")
        #assumptions: engPOS and filPOS are strings that contains POS tags sperated by -
        tokenizedText = nltk.word_tokenize(post.content)

        #step 1: split tweets/posts into sentences
        text = ' '.join(tokenizedText)
        finalPOSTags = []
        filTags =[]

        engTags = [tag for tag in post.epos.split("-")]

        splitFPOS = post.fpos.split("-")
        for i in range(len(splitFPOS)):
            if(i != len(splitFPOS)-1):

                tag = splitFPOS[i] +" "+splitFPOS[i+1]
                if (self.mapping.get(tag) != None): #check if compound tag is in the map
                    i = i+1
                    filTags.append(tag)
                else:
                    filTags.append(self.mapping.get(splitFPOS[i],splitFPOS[i]))
            else :
                filTags.append(self.mapping.get(splitFPOS[i], splitFPOS[i]))

        startIndex = 0

        print("resulting mapped fil tags:", filTags)
        #step 2: detect the language of each sentence
        langDetector = LanguageDetector()

        for sentence in nltk.sent_tokenize(text):
            language = langDetector.getLanguage(sentence)
            wordCount = sentence.split()

            if(language == Language.FILIPINO):
                for i in range(startIndex,len(wordCount) + startIndex):
                    finalPOSTags.append(filTags[i])

            elif(language == Language.ENGLISH):
                for i in range(startIndex,len(wordCount) + startIndex):
                    finalPOSTags.append(engTags[i])

            else:
                for i in range(startIndex,len(wordCount) + startIndex):

                    if (engTags[i] == filTags[i]):
                        finalPOSTags.append(engTags[i])

                    elif (filTags[i] == 'UNK'):
                        finalPOSTags.append(engTags[i])

                    elif (filTags[i] == 'FW' and engTags[i] != 'FW'):
                        finalPOSTags.append(engTags[i])

                    elif (engTags[i] == 'FW' and filTags[i] != 'FW'):
                        finalPOSTags.append(filTags[i])

                    elif (filTags[i][:2] == engTags[i][:2]):
                        finalPOSTags.append(filTags[i][:2])

                    else:
                        finalPOSTags.append(filTags[i])

           # print("counter: ",counter)
            startIndex = startIndex + len(wordCount)
            print("finalPOSTag size:", len(finalPOSTags))

        #self.posDAO.updateCombinedPOS(post.id, finalPOSTags)
        print(engTags)
        print(filTags)
        return '-'.join(finalPOSTags)

