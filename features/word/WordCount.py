import nltk
import re
from collections import Counter

class WordCount:

    ABBREVIATIONS_FILENAME = 'abbreviations.txt'

    def __init__(self):
        with open(self.ABBREVIATIONS_FILENAME) as f:
            ABBREVIATIONS = f.read().splitlines()
        self.ABBREVIATIONS = [abbrev.lower() for abbrev in ABBREVIATIONS]

    def getTotalNumberOfWords(self, text):
        nWords = 0
        words = nltk.word_tokenize(text)
        for word in words:
            if word[0].isalnum():
                nWords += 1
        return nWords

    def getNWordsBegCapital(self, text):
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            if word[0].isupper():
                count += 1
        return count

    def getAveLengthWords(self, text):
        aveLength = 0
        nWords = 0
        words = nltk.word_tokenize(text)
        for word in words:
            if word[0].isalnum():
                aveLength += len(word)
                nWords += 1
        return aveLength/nWords

    def getNWordsWithRepLetters(self, text):
        nWords = 0
        words = nltk.word_tokenize(text)
        for word in words:
            if word[0].isalnum():
                word = word.lower()
                i = 0
                stat = False
                while i < len(word)-2 and stat != True:
                    prevLetter = word[i]
                    count = 1
                    j = i + 1
                    while j < len(word) and stat != True:
                        if prevLetter == word[j]:
                            count += 1
                        if count > 2:
                            nWords += 1;
                            stat = True
                        j += 1
                    i += 1
        return nWords

    def getRatioOfUniqueWords(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            word = word.lower()
            if word[0].isalnum():
                legitWords.append(word)
        return len(set(legitWords))/len(legitWords)

    def getRatioOfShortWords(self, text):
        words = nltk.word_tokenize(text)
        nShortWords = 0
        nWords = 0
        for word in words:
            if word[0].isalnum():
                nWords += 1
                if len(word) <= 3:
                    nShortWords += 1
        return nShortWords/nWords

    def getLolHmmCount(self, text):
        count = 0
        lolPattern = re.compile('l+o+l+')
        hmmPattern = re.compile('h+m+')
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if lolPattern.match(word) or hmmPattern.match(word):
                count += 1
        return count

    def getHapaxLegomena(self, text):
        onceWords = []
        nWords = 0
        words = nltk.word_tokenize(text)
        legitWords = []
        for i in range(len(words)):
            word = words[i].lower()
            if word[0].isalnum():
                nWords += 1
                legitWords.append(word)
        counted = dict(Counter(legitWords))
        print(counted)
        for key in counted:
            if counted[key] == 1:
                onceWords.append(key)
        return len(onceWords)/nWords

    def getHapaxDislegomena(self, text):
        twiceWords = []
        nWords = 0
        words = nltk.word_tokenize(text)
        legitWords = []
        for i in range(len(words)):
            word = words[i].lower()
            if word[0].isalnum():
                nWords += 1
                legitWords.append(word)
        counted = dict(Counter(legitWords))
        print(counted)
        for key in counted:
            if counted[key] == 2:
                twiceWords.append(key)
        return len(twiceWords) / nWords

    def getWordLengthFreqDist(self, text):
        nWords = 0
        freqDist = [0 for i in range(20)]
        words = nltk.word_tokenize(text)
        legitWords = []
        for i in range(len(words)):
            word = words[i].lower()
            if word[0].isalnum():
                if len(word) <= 20:
                    freqDist[len(word)-1] += 1
                nWords += 1
        return freqDist

    def getRatioOfNetAbbrev(self, text):
        nWords = 0
        nAbbrevs = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word[0].isalnum():
                nWords += 1
                if word in self.ABBREVIATIONS:
                    nAbbrevs += 1
        return nAbbrevs/nWords