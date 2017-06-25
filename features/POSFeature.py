from collections import Counter

import jpype
from jpype import *
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from utility.LanguageDetector import LanguageDetector, Language
from utility.PostCleaner import PostCleaner
import time

class POSFeature:
    VERB = 'VB'
    ADJECTIVE = 'JJ'
    UNKNOWN = 'UNK'

    def __init__(self):
        self.nVerbs = 0
        self.nAdjectives = 0
        self.sPOS = ''
        self.mapping = {}
        self.postCleaner = PostCleaner()

    def populateMappingDictionary(self):
        print("populating map dictionary...")
        with open("mapping.txt", "r") as file_object:
            for line in file_object:
                splitline = line.split()
                self.mapping[' '.join(splitline[1:])] = splitline[0]

    def getEnglishPOS(self, text):
        '''jvmPath = jpype.getDefaultJVMPath()
        jpype.startJVM(jvmPath, "-Djava.class.path=dependencies/NormAPI.jar;dependencies/RBPOST.jar")
        rbpost = JPackage("rbpost").RBPOST

        result = rbpost.tokenizer_Text(text)
        tokenizedText = result.split(" ")
        jpype.shutdownJVM()'''


        tokenizedText = nltk.word_tokenize(text)
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


    def getCombinedPOSTag(self, post):

        print("combining pos...[",post.id,"]")
        #assumptions: engPOS and filPOS are strings that contains POS tags separated by -
        #step 1: split tweets/posts into sentences and mapped fpos to epos

        text = post.content
        text = self.postCleaner.changeEmojisToText(text)
        text = self.postCleaner.normalizeUnicode(text)
        text = self.postCleaner.changeForeignToText(text)
        text = self.postCleaner.changeLinkToText(text)
        text = ' '.join(nltk.word_tokenize(text))

        finalPOSTags = []
        filTags =[]

        #step 1.1: get the resulting epos tag
        engTags = [tag for tag in post.epos.split("-")]

        #step 1.2: get the resulting fpos tag
        splitFPOS = post.fpos.split("-")

        #print("raw eng tags:", engTags)

        #step 1.3: mapped the fpos to its corresponding epos
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

        #print("map fil tags:", filTags)

        #step 2: detect the language of each sentence
        langDetector = LanguageDetector()

        #print("sentence list:", nltk.sent_tokenize(text))
        for sentence in nltk.sent_tokenize(text):
            language = langDetector.getLanguage(sentence)
            wordCount = len(sentence.split())
            #print("word count:", wordCount , " sentence:", sentence, " language:", language)
            if(language == Language.FILIPINO):
                for i in range(startIndex, wordCount + startIndex):
                    finalPOSTags.append(filTags[i])
            elif(language == Language.ENGLISH):
                for i in range(startIndex, wordCount + startIndex):
                    finalPOSTags.append(engTags[i])
            else:
                if(len(engTags) != len(filTags)):
                    for i in range(startIndex, wordCount + startIndex):
                        finalPOSTags.append(engTags[i])
                    print("not same length...............................[",wordCount,"] ", sentence)
                    #time.sleep(5.5)
                else:
                    for i in range(startIndex, wordCount + startIndex):
                       # print("index[",i,"] range:", wordCount +startIndex)

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

            startIndex = startIndex + wordCount
        #print("final result:",finalPOSTags)
        return '-'.join(finalPOSTags)


