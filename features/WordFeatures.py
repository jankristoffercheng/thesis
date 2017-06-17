import string
import re


def getTotalNumberOfCharacters(text):
    return len(text)

def getTotalNumberOfLetters(text):
    count = 0
    for c in text:
        if c.isalnum():
            count += 1
    return count

def getTotalNumberOfUppercase(text):
    count = 0
    for c in text:
        if c.isalpha():
            if c.isupper():
                count += 1
    return count

def getTotalNumberOfDigitalNumbers(text):
    count = 0
    for c in text:
        if c.isnumeric():
                count += 1
    return count

def getNumberOfWhiteSpaces(text):
    count = 0
    for c in text:
        if c.isspace():
                count += 1
    return count

def getNumberOfSpecialChars(text):
    count=0
    invalidChars = set(string.punctuation.replace("!", "").replace(".","").replace("?",""))
    for c in text:
        if c in invalidChars:
            count+=1
    return count

def getFrequencyOfAbbreviations(text):
    count =0
    return count

def getNumberOfRepetitiveAlphaCharacters(text):
    #list = re.findall(r'(([a-zA-Z]){2,})',text)
    #print(list)
    list = re.findall(r'(([a-zA-Z])\2{2,})',text)
    print(list)
    num = (len(n) for n,z in list)
    print(len(list), sum(num)/len(list))
    return len(list), sum(num)/len(list)

def getNumberOfRepeatedPunctuationMarks(text):
    list = re.findall(r'(([!?.]){2,})',text)
    print(list)
    #list = re.findall(r'(([!?.])\2{2,})', text)
    #print(list)
    print(list)
    num = (len(n) for n,z in list)
    print(len(list), sum(num)/len(list))
    return list

getNumberOfRepetitiveAlphaCharacters('http://www.google.com/search=ooo-jjjj-1111-!!!-....-???-...')
getNumberOfRepeatedPunctuationMarks('http://www.google.com/search=ooo-jjjj-1111-!!!-....-???-...-!.!.?!-.!')
