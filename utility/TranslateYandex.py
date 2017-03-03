from yandex_translate import YandexTranslate

class TranslateYandex:

    def __init__(self):
        self.translate = YandexTranslate('trnsl.1.1.20170225T080044Z.7a0894099424d9cf.24ff5ce7f70e0b9adeca2c5aac006b98d9c9fc29')

    def translateText(self, text):
        try:
            result = self.translate.translate(text, 'tl-en')
            return result['text'][0]
        except:
            return "ERROR"