from collections import Counter

import jpype
from jpype import *
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from utility.LanguageDetector import LanguageDetector, Language
from utility.PostCleaner import PostCleaner
import time

class POSFeature:

    def __init__(self):
        self.sPOS = ''
        self.mapping = {}
        self.postCleaner = PostCleaner()

    def populateMappingDictionary(self):
        """
            This method populates the dictionary to include the list of mapped
            Filipino POS to its equivalent English POS by reading
            the contents of the mapping.txt file
        """
        with open("mapping.txt", "r") as file_object:
            for line in file_object:
                splitline = line.split()
                self.mapping[' '.join(splitline[1:])] = splitline[0]

    def getEnglishPOS(self, text):
        """
        :param text: text to be processed
        :return: returns a string of the resulting POS tags joined by "-"
        """
        tokenizedText = nltk.word_tokenize(text)
        pos = nltk.pos_tag(tokenizedText)
        pos = '-'.join([posTag[1] for posTag in pos])
        return pos

    def getCombinedPOSTag(self, post):
        """
            This method combines the resulting English and Filipino POS tags from the two separate POS Tagger.
            :param post: the document to be processed
            :return: returns a string of combined POS tags joined by "-"
        """

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

        #step 2: detect the language of each sentence
        langDetector = LanguageDetector()

        #print("sentence list:", nltk.sent_tokenize(text))
        for sentence in nltk.sent_tokenize(text):
            language = langDetector.getLanguage(sentence)
            wordCount = len(sentence.split())

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

                else:
                    for i in range(startIndex, wordCount + startIndex):

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
        return '-'.join(finalPOSTags)


