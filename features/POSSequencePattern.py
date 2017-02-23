from collections import Counter
import nltk
import math

from model.Document import Document


class POSSequencePattern:

    MAX_LENGTH = 7

    def __init__(self, documentList):
        self.documentList = documentList
        self.tagList = []
        self.cList = []
        self.fList = []
        self.retrievePOSTags_docFrequecy()

    def retrievePOSTags_docFrequecy(self):

        tagDict = {}
        for document in self.documentList:
            #tag[0] = word
            #tag[1] = POS equivalent
            print("POS: ",document.posSequence)
            for tag in list(set(document.posSequence.split('-'))):
                tagDict[tag] = tagDict[tag] + 1 if tag in tagDict else 1

            self.cList.append(tagDict)

        self.tagList = tagDict.keys()

    def minePOSPatterns(self, minsup, minadherence):

        self.fList.append({tag: count for tag, count in self.cList[0].items() if count / len(self.documentList) >= minsup})
        sequencePatterns = []
        sequencePatterns.extend([key for key in self.fList[0]])

        for i in range(1,self.MAX_LENGTH):
            C = self.candidateGen(self.fList[i-1])
            for document in self.documentList:
                for key in C:
                    value = C[key]
                    sublist = key.split('-')
                    list = document.posSequence.split('-')
                    n = len(sublist)
                    if any((sublist == list[k:k + n]) for k in range(len(list) - n + 1)):
                        C[key] += 1

            self.fList.append({tag: count for tag, count in C.items() if count / len(self.documentList) >= minsup})
            for key in self.fList[-1]:
                value = self.fList[-1][key]
                if self.computeFairSCP(key, value) >= minadherence:
                    sequencePatterns.append(key)

        return sequencePatterns



    def computeFairSCP(self, key, count):
        sumResult = 0
        tagSplit = key.split("-")
        seqLen = len(tagSplit)

        for i in range(0,seqLen-1):

            posTagged = ['-' + tagSplit[k] for k in range(0, i+1)]
            pos1 = ''.join(posTagged)[1:]
            pos2 = pos1[len(pos1):]
            temp = 0
            for d in self.documentList:
                if pos2 in d.content:
                    temp += 1
            sumResult += self.fList[len(pos1.split('-'))-1][pos1] * temp

        return math.pow(count, 2)/(1/(seqLen-1) * sumResult)


    def candidateGen(self, fList):
        countPOS = {}
        for key in fList:
            value = fList[key]
            for tag in self.tagList:
                tag_suffix = key +"-"+ tag
                countPOS[tag_suffix] = 0

        return countPOS

#document = []
#document.append("Start a shit a")
#document.append("Jump the rope a")
#document.append("Create shit blog a")

#p = POSSequencePattern(document)
#print("RESULT: ",p.minePOSPatterns(0.30, 0.2))