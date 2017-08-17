import nltk
import re

class FunctionWordCount:

    FUNCTIONWORDS_FILENAME = 'features/functionwords.txt'

    def __init__(self):
        """
            initializes the list of function words from a text file
        """
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
        """
            :param text: string to be counted for articles
            :return: total number of articles
        """
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
        """
            :param text: string to be counted for pro-sentence words
            :return: total number of pro-sentence words
        """
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
        """
            :param text: string to be counted for pronouns
            :return: total number of pronouns
        """
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
        """
            :param text: string to be counted for auxillary words
            :return: total number of auxillary words
        """
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
        """
            :param text: string to be counted for conjunctions
            :return: total number of conjunctions
        """
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
        """
            :param text: string to be counted for interjections
            :return: total number of interjections
        """
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
        """
            :param text: string to be counted for adposition words
            :return: total number of adposition words
        """
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
        """
            :param text: string to be counted for all function words
            :return: total number of all function words
        """
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





