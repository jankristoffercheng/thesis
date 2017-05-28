import nltk

class FunctionWordCount:

    FUNCTIONWORDS_FILENAME = 'functionwords.txt'

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
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.ARTICLES:
                count += 1
        return count

    def getProSentenceCount(self, text):
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.PROSENTENCE:
                count += 1
        return count

    def getPronounCount(self, text):
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.PRONOUNS:
                count += 1
        return count

    def getAuxillaryCount(self, text):
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.AUXILLARY:
                count += 1
        return count

    def getConjunctionCount(self, text):
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.CONJUNCTION:
                count += 1
        return count

    def getInterjectionCount(self, text):
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.INTERJECTION:
                count += 1
        return count

    def getAdpositionCount(self, text):
        count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.ADPOSITION:
                count += 1
        return count

    def getAllFunctionWordCount(self, text):
        totalCount = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.ALLFUNCTIONWORDS:
                #print(word)
                totalCount += 1
        return totalCount
