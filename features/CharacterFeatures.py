import string
import re

class CharacterFeatures:
    """
    Returns the character features of a text
    """
    def getTotalNumberOfCharacters(self, text):
        """
        :param text: text to be processed
        :return: total number ofcharacters
        """
        return len(text)

    def getTotalNumberOfLetters(self, text):
        """
        :param text: text to be processed
        :return: total number of letters
        """
        count = 0
        for c in text:
            if c.isalnum():
                count += 1
        return count

    def getTotalNumberOfUppercase(self, text):
        """
        :param text: text to be processed
        :return: total number of uppercase letters
        """
        count = 0
        for c in text:
            if c.isalpha():
                if c.isupper():
                    count += 1
        return count

    def getTotalNumberOfDigitalNumbers(self, text):
        """
        :param text: text to be processed
        :return: total number of digital numbers
        """
        count = 0
        for c in text:
            if c.isnumeric():
                    count += 1
        return count

    def getNumberOfWhiteSpaces(self, text):
        """
        :param text: text to be processed
        :return: total number of white spaces
        """
        count = 0
        for c in text:
            if c.isspace():
                    count += 1
        return count

    def getNumberOfSpecialChars(self, text):
        """
        :param text: text to be processed
        :return: total number of special characters besides punctuation marks
        """
        count=0
        invalidChars = set(string.punctuation.replace("!", "").replace(".","").replace("?",""))
        for c in text:
            if c in invalidChars:
                count+=1
        return count

    def getNumberOfRepetitiveAlphaCharacters(self, text):
        """
        :param text: text to be processed
        :return: total number of instances that alpha characters are repeated more than twice consecutively
        """
        #list = re.findall(r'(([a-zA-Z]){2,})',text)
        #print(list)
        list = re.findall(r'(([a-zA-Z])\2{2,})',text)
        print(list)
        num = (len(n) for n,z in list)
        # print(len(list), sum(num)/len(list))
        return len(list)

    def getNumberOfRepeatedPunctuationMarks(self, text):
        """
        :param text: text to be processed
        :return: total number of instances of consecutive punctuation marks
        """
        list = re.findall(r'(([!?.]){2,})',text)
        print(list)
        #list = re.findall(r'(([!?.])\2{2,})', text)
        #print(list)
        print(list)
        num = (len(n) for n,z in list)
        # print(len(list), sum(num)/len(list))
        return len(list)

# CharacterFeatures().getNumberOfRepetitiveAlphaCharacters('http://www.google.com/search=ooo-jjjj-1111-!!!-....-???-...')
# CharacterFeatures().getNumberOfRepeatedPunctuationMarks('http://www.google.com/search=ooo-jjjj-1111-!!!-....-???-...-!.!.?!-.!')
