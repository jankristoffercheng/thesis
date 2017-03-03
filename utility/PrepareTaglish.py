from utility.LanguageDetector import LanguageDetector
from utility.NormalizeFilipino import NormalizeFilipino
from utility.TranslateYandex import TranslateYandex

class PrepareTaglish:

    def __init__(self):
        self.languageDetector = LanguageDetector()
        self.filipinoNormalizer = NormalizeFilipino()
        self.translator = TranslateYandex()

    def testLanguageAndTranslate(self, text):
        language = self.languageDetector.getLanguage(text)
        if(language != 2):
            return ""

        normalized = self.filipinoNormalizer.normalize_Filipino(text)
        translated = self.translator.translateText(normalized)

        return translated


print(PrepareTaglish().getEnglishOf("I am so happy about this. These are English words. Mahilig aq sa pagkain. nand2 k n b? "))