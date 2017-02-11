from langdetect import detect_langs
from langdetect import DetectorFactory
from langdetect import lang_detect_exception

class Language:
    UNKNOWN = -1
    ENGLISH = 0
    FILIPINO = 1
    TAGLISH = 2

    # method in case you want to decipher the codes :)
    def getLanguage(self, code):
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

        try:
            print(detect_langs(text))
        except:
            print("UNKNOWN")

    def englishOrTagalog(self, string):
        res = detect_langs(string)
        for item in res:
            if item.lang == "tl" or item.lang == "en":
                return item.lang
        return None


## trial
langdetect = LanguageDetector()
language = Language()

print(language.getLanguage(langdetect.getLanguage("Maglalagay sana ako ng caption na malalim kaso di ako artist.")))
print(language.getLanguage(langdetect.getLanguage("!!!")))
print(language.getLanguage(langdetect.getLanguage("Dahil puno na sa restaurant, mom, shoti, and I were asked to share a seat with a customer. Buti nalang ikaw kasama namin sa table. Nice seeing you Shayane.")))
print(language.getLanguage(langdetect.getLanguage("Ahhhh tangina puso ko. Tama ka Joon Hyung my heart is fckn flattering")))
langdetect.englishOrTagalog("ampotek hahaha")
langdetect.getLanguageDetailed(("ahhhh tangina puso ko. tama ka Joon Hyung my heart is fckn flattering.").lower())