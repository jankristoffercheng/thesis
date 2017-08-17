from langdetect import detect_langs
from langdetect import DetectorFactory
from langdetect import lang_detect_exception



class Language:
    UNKNOWN = -1
    ENGLISH = 0
    FILIPINO = 1
    TAGLISH = 2

    def getLanguage(self, code):
        """
            :param code: integer assigned to represent a language
            :return: the meaning of the codes
        """
        if code == -1:
            return "UNKNOWN"
        elif code == 0:
            return "ENGLISH"
        elif code == 1:
            return "FILIPINO"
        elif code == 2:
            return "TAGLISH"

class LanguageDetector:

    def __init__(self):
        DetectorFactory.seed = 0  # to enforce consistent results

    def getLanguage(self, text):
        """
            :param text: string to be language detected
            :return: "ENGLISH", "FILIPINO" or "TAGALOG", else "UNKNOWN"
        """
        try:
            result = detect_langs(text)[0]

            if result.prob >= 0.90 and result.lang == "en":
                return Language.ENGLISH
            elif result.prob >= 0.90 and result.lang == "tl":
                return Language.FILIPINO
            elif result.lang == "tl" or result.lang == "en":
                return Language.TAGLISH
            else:
                return Language.UNKNOWN

        except:
            return Language.UNKNOWN

    def getLanguageDetailed(self, text):
        """
            :param text: string to be language identified
            :return: detailed probabilities of the languages detected, else "UNKNOWN"
        """
        try:
            print(detect_langs(text))
        except:
            print("UNKNOWN")

    def englishOrTagalog(self, string):
        """
            :param string: string to be identified as either English or Tagalog
            :return: strings of "en" (English) or "tl" (Tagalog)
        """
        res = detect_langs(string)
        for item in res:
            if item.lang == "tl" or item.lang == "en":
                return item.lang
        return None

