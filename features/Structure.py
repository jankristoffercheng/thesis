from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from features.WordCount import WordCount
import nltk
class Structure:

    ABBREVIATIONS_FILENAME = 'features/abbreviations.txt'
    def __init__(self):
        with open(self.ABBREVIATIONS_FILENAME) as f:
            ABBREVIATIONS = f.read().splitlines()
        self.ABBREVIATIONS = [abbrev.lower() for abbrev in ABBREVIATIONS]
        self.punkt_param = PunktParameters()
        self.punkt_param.abbrev_types = set(self.ABBREVIATIONS)
        self.tokenizer = PunktSentenceTokenizer(self.punkt_param)
        self.wordCounter = WordCount()

    def getNSentences(self,text):
        """
        :param text: text to be processed
        :return: returns an integer of the number of sentences detected on the text
        """
        return len(self.tokenizer.tokenize(text))

    def getAvgNWordPerSentence(self, text):
        """
        :param text: text to be processed
        :return: returns a float of the average number of words per sentences detected on the text
        """
        nWords = self.wordCounter.getTotalNumberOfWords(text)
        nSentences = self.getNSentences(text)
        if nSentences == 0:
            return 0
        return nWords/nSentences

    def getNSentenceBegUpper(self,text):
        """
        :param text: text to be processed
        :return: returns an integer on the number of sentences beginning with an uppercase.
        """
        nCount = 0
        for sentence in self.tokenizer.tokenize(text):
            words = nltk.word_tokenize(sentence)
            if words[0][0].isupper():
                nCount += 1
        return nCount

    def getNSentenceBegLower(self,text):
        """
        :param text: text to be processed
        :return: returns an integer on the number of sentences beginning with an lowercase.
        """
        nCount = 0
        for sentence in self.tokenizer.tokenize(text):
            words = nltk.word_tokenize(sentence)
            if words[0][0].islower():
                nCount += 1
        return nCount

    def getParagraphs(self, text):
        """
        :param text: text to be processed
        :return: returns a list containing the detected paragraphs
        """
        return text.split('\n')

    def getNParagraphs(self, text):
        """

        :param text: text to be processed
        :return: returns an integer of the number of detected paragraphs on the text.
        """
        return len(self.getParagraphs(text))

    def getAvgNSentencePerParagraph(self,text):
        """
        :param text: text to be processed
        :return: returns a float of the average number of sentences per paragraphs detected on the text
        """
        nSentences = self.getNSentences(text)
        nParagraphs = self.getNParagraphs(text)
        if nParagraphs == 0:
            return 0
        return nSentences/nParagraphs

    def getAvgNWordPerParagraph(self,text):
        """
        :param text: text to be processed
        :return: returns a float on the average number of words per paragraphs detected on the text
        """
        nWords = self.wordCounter.getTotalNumberOfWords(text)
        nParagraphs = self.getNParagraphs(text)
        if nParagraphs == 0:
            return 0
        return nWords/nParagraphs

    #not sure if correct implementation
    #are spaces counted/considered?
    def getAvgNCharacterPerParagraph(self,text):
        """
        :param text: text to be processed
        :return: returns a float of the average number of characters per paragaphs detected on the text
        """
        nChar = len(text)
        nParagraphs = self.getNParagraphs(text)

        return nChar/nParagraphs
