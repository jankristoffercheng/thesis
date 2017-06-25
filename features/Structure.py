from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from features.WordCount import WordCount
import nltk
class Structure:

    ABBREVIATIONS_FILENAME = '../features/abbreviations.txt'
    def __init__(self):
        with open(self.ABBREVIATIONS_FILENAME) as f:
            ABBREVIATIONS = f.read().splitlines()
        self.ABBREVIATIONS = [abbrev.lower() for abbrev in ABBREVIATIONS]
        self.punkt_param = PunktParameters()
        self.punkt_param.abbrev_types = set(self.ABBREVIATIONS)
        self.tokenizer = PunktSentenceTokenizer(self.punkt_param)
        self.wordCounter = WordCount()

    def getNSentences(self,text):
        return len(self.tokenizer.tokenize(text))

    def getAvgNWordPerSentence(self, text):
        nWords = self.wordCounter.getTotalNumberOfWords(text)
        nSentences = self.getNSentences(text)

        return nWords/nSentences

    def getNSentenceBegUpper(self,text):
        nCount = 0
        for sentence in self.tokenizer.tokenize(text):
            words = nltk.word_tokenize(sentence)
            if words[0][0].isupper():
                nCount += 1
        return nCount

    def getNSentenceBegLower(self,text):
        nCount = 0
        for sentence in self.tokenizer.tokenize(text):
            words = nltk.word_tokenize(sentence)
            if words[0][0].islower():
                nCount += 1
        return nCount

    def getParagraphs(self, text):
        return text.split('\n')

    def getNParagraphs(self, text):
        return len(self.getParagraphs(text))

    def getAvgNSentencePerParagraph(self,text):
        nSentences = self.getNSentences(text)
        nParagraphs = self.getNParagraphs(text)

        return nSentences/nParagraphs

    def getAvgNWordPerParagraph(self,text):
        nWords = self.wordCounter.getTotalNumberOfWords(text)
        nParagraphs = self.getNParagraphs(text)

        return nWords/nParagraphs

    #not sure if correct implementation
    #are spaces counted/considered?
    def getAvgNCharacterPerParagraph(self,text):
        nChar = len(text)
        nParagraphs = self.getNParagraphs(text)

        return nChar/nParagraphs

# structure = Structure()
# print(structure.getNSentenceBegUpper("Nag-aral nang masaya ang mga mag-aaral. pupunta sa paaralan ang mga mag-aaral...dito na ba ?"))
'''
text = "Hi my name is shayane! \n do you kniw me?"
print(text)
posDAO = PostsDAO()
posDAO.addPost("00031", text, "05", "30")
'''
