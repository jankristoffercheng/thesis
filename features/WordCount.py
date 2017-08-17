import nltk
import math
import re
from collections import Counter

class WordCount:

    ABBREVIATIONS_FILENAME = 'features/abbreviations.txt'

    def __init__(self):
        """
        initializes the list of abbreviations from a text file
        """
        with open(self.ABBREVIATIONS_FILENAME) as f:
            ABBREVIATIONS = f.read().splitlines()
        self.ABBREVIATIONS = [abbrev.lower() for abbrev in ABBREVIATIONS]

    def getTotalNumberOfWords(self, text):
        """
        :param text: string to be used
        :return: total number of words
        """
        nWords = 0
        words = nltk.word_tokenize(text)
        for word in words:
            if word[0].isalnum():
                nWords += 1
        return nWords

    def getNWordsBegCapital(self, text):
        """
        :param text: string to be used
        :return: number of words beginning with a capital letter
        """
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            if word[0].isupper():
                count += 1
        return count

    def getAveLengthWords(self, text):
        """
        :param text: string to be used
        :return: average length of words
        """
        aveLength = 0
        nWords = 0
        words = nltk.word_tokenize(text)
        for word in words:
            if word[0].isalnum():
                aveLength += len(word)
                nWords  += 1
        if(nWords==0): return  0
        else: return aveLength/nWords

    def getNWordsWithRepLetters(self, text):
        """
        :param text: string to be used
        :return: number of words with repeating letters
        """
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
        """
        :param text: string to be used
        :return: ratio of unique words to total number of words
        """
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            word = word.lower()
            if word[0].isalnum():
                legitWords.append(word)
        if(len(legitWords)==0): return 0
        else: return len(set(legitWords))/len(legitWords)

    def getRatioOfShortWords(self, text):
        """
        :param text: string to be used
        :return: ratio of words with less than 3 characters to total number of words
        """
        words = nltk.word_tokenize(text)
        nShortWords = 0
        nWords = 0
        for word in words:
            if word[0].isalnum():
                nWords += 1
                if len(word) <= 3:
                    nShortWords += 1
        if(nWords==0): return 0
        else: return nShortWords/nWords

    def getLolHmmCount(self, text):
        """
        :param text: string to be used
        :return: number of lol's and hmm's with the use of regex
        """
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
        """
        :param text: string to be used
        :return: hapax legomena
        """
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
        for key in counted:
            if counted[key] == 1:
                onceWords.append(key)
        return len(onceWords)

    def getHapaxDislegomena(self, text):
        """
        :param text: string to be used
        :return: hapax dislegomena
        """
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
        for key in counted:
            if counted[key] == 2:
                twiceWords.append(key)
        return len(twiceWords)

    def getRatioOfHapaxLegomena(self, text):
        """
        :param text: string to be used
        :return: ratio of hapax legomena to total number of words
        """
        if (self.getTotalNumberOfWords(text) == 0):
            return 0
        else:
            return self.getHapaxLegomena(text)/self.getTotalNumberOfWords(text)

    def getRatioOfHapaxDislegomena(self, text):
        """
        :param text: string to be used
        :return: ratio of hapax dislegomena to total number of words
        """
        if (self.getTotalNumberOfWords(text) == 0):
            return 0
        else:
            return self.getHapaxDislegomena(text)/self.getTotalNumberOfWords(text)

    def getWordLengthFreqDist(self, text):
        """
        :param text: string to be used
        :return: an array with the word length frequency distribution from length 1 to 20
        """
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
        """
        :param text: string to be used
        :return: ratio of net abbriavtions to toal number of words
        """
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.ABBREVIATIONS]
        abbrevs = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)
        if (self.getTotalNumberOfWords(text) == 0):
            return 0
        else:
            return len(abbrevs)/self.getTotalNumberOfWords(text)

    def getDictOfWordsMappedToOccurrence(self, text):
        """
        :param text: string to be used
        :return: dictionary of words mapped to occurrence
        """
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        wordDict = dict(Counter(legitWords))
        return wordDict

    def getOccurrenceArray(self, text):
        """
        :param text: string to be used
        :return: array of word occurrences
        """
        wordDict = self.getDictOfWordsMappedToOccurrence(text)
        if(len(wordDict)>0):
            maxKey = max(wordDict, key=wordDict.get)
            maxVal = wordDict[maxKey]
            occurrenceA = [0 for i in range(maxVal)]
            for key, val in wordDict.items():
                occurrenceA[val - 1] += 1
            return occurrenceA
        else: return []

    def getYulesK(self, text):
        """
        :param text: string to be used
        :return: yules k measure
        """
        summation = 0
        occurrenceA = self.getOccurrenceArray(text)
        N = self.getTotalNumberOfWords(text)
        for i in range(len(occurrenceA)):
            summation += occurrenceA[i] * ((i+1)/N) * ((i+1)/N)
        if (N == 0):
            return 0
        else:
            return 10000 * ((-1/N) + summation)

    def getSimpsonsD(self, text):
        """
        :param text: string to be used
        :return: simpsons d
        """
        summation = 0
        occurrenceA = self.getOccurrenceArray(text)
        N = self.getTotalNumberOfWords(text)
        for i in range(len(occurrenceA)):
            summation += occurrenceA[i] * ((i + 1) / N) * (i/N-1)
        return summation

    def getNDifferentWords(self, text):
        """
        :param text: string to be used
        :return: number of unique words
        """
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        return len(set(legitWords))

    def getSichelsS(self, text):
        """
        :param text: string to be used
        :return: sichels s
        """
        if(self.getNDifferentWords(text)>0): return self.getHapaxDislegomena(text)/self.getNDifferentWords(text)
        else: return 0

    def getHonoresR(self, text):
        """
        :param text: string to be used
        :return: honores r
        """
        N = self.getTotalNumberOfWords(text)
        V = self.getNDifferentWords(text)
        hapaxLegomena = self.getHapaxLegomena(text)
        if(V!=0 and (1 - (hapaxLegomena/V))!=0): return 100 * math.log10(N) / (1 - (hapaxLegomena/V))
        else: return 0

    def getEntropy(self, text):
        """
        :param text: string to be used
        :return: entropy
        """
        summation = 0
        occurrenceA = self.getOccurrenceArray(text)
        N = self.getTotalNumberOfWords(text)
        for i in range(len(occurrenceA)):
            summation += occurrenceA[i] * (-math.log10((i+1)/N)) * ((i+1)/N)
        return summation