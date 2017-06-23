import nltk
import re

class FunctionWordCount:

    FUNCTIONWORDS_FILENAME = '../features/functionwords/functionwords.txt'

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
        '''count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.ARTICLES:
                count += 1
        return count'''
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
        '''count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.PROSENTENCE:
                count += 1
        return count'''
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
        '''count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.PRONOUNS:
                count += 1
        return count'''
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
        '''count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.AUXILLARY:
                count += 1
        return count'''
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
        '''count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.CONJUNCTION:
                count += 1
        return count'''
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
        '''count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.INTERJECTION:
                count += 1
        return count'''
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
        '''count = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.ADPOSITION:
                count += 1
        return count'''
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
        '''totalCount = 0
        words = nltk.word_tokenize(text)
        for word in words:
            word = word.lower()
            if word in self.ALLFUNCTIONWORDS:
                #print(word)
                totalCount += 1
        return totalCount'''
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

#print(FunctionWordCount().getAllFunctionWordCount("aboard,astride,down,of,through,worth,on to,in front of,about,at,during,off,throughout,according to,onto,in lieu of,above,athwart,except,on,till,ahead to,out fro,in place of,absent,atop,failing,onto,to,as to,out of,in spite of,across,barring,following,opposite,toward,aside from,outside of,on account of,after,before,for,out,towards,because of,owing to,on behalf of,against,behind,from,outside,under,close to,prior to,on top of,along,below,in,over,underneath,due to,pursuant to,versus,alongside,beneath,inside,past,unlike,except for,regardless of,concerning,amid,beside,into,per,until,far from,subsequent to,considering,amidst,besides,like,plus,up,in to,as far as,regarding,among,between,mid,regarding,upon,into,as well as,apart from,amongst,beyond,minus,round,via,inside of,by means of,around,but,near,save,with,instead of,in accordance with,as,by,next,since,within,near to,in addition to,aslant,despite,notwithstanding,than,without,next to,in case of"))