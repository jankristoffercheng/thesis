import nltk
import re

class FunctionWordCount:

    FUNCTIONWORDS_FILENAME = 'features/functionwords.txt'

    def __init__(self):
        with open(self.FUNCTIONWORDS_FILENAME) as f:
            list = f.read().splitlines()
        self.ARTICLES = list[0].split(",")
        self.PROSENTENCE = list[1].split(",")
        self.PRONOUNS = list[2].split(",")
        self.AUXILLARY = list[3].split(",")
        self.CONJUNCTION = list[4].split(",")
        self.INTERJECTION = list[5].split(",")
        self.ADPOSITION = list[6].split(",")
        self.ALLFUNCTIONWORDS = self.ARTICLES + self.PROSENTENCE + self.PRONOUNS + self.AUXILLARY + self.CONJUNCTION + self.INTERJECTION + self.ADPOSITION

    def getArticleCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.ARTICLES]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)

        return len(functionwords)

    def getProSentenceCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.PROSENTENCE]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)

        return len(functionwords)

    def getPronounCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.PRONOUNS]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)

        return len(functionwords)

    def getAuxillaryCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.AUXILLARY]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)

        return len(functionwords)

    def getConjunctionCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.CONJUNCTION]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)

        return len(functionwords)

    def getInterjectionCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.INTERJECTION]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)

        return len(functionwords)

    def getAdpositionCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.ADPOSITION]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)

        return len(functionwords)

    def getAllFunctionWordCount(self, text):
        words = nltk.word_tokenize(text)
        legitWords = []
        for word in words:
            if word[0].isalnum():
                legitWords.append(word)
        text = ' '.join(legitWords)
        regexes = ['(' + '+'.join(abbrev) + '+)' for abbrev in self.ALLFUNCTIONWORDS]
        functionwords = re.findall('\\b(' + '|'.join(regexes) + ')\\b', text)
        print(functionwords)

        return len(functionwords)





