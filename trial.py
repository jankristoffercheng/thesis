'''from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory

DetectorFactory.seed = 0

print(detect_langs("Maglalagay sana ako ng caption na malalim kaso di ako artist."))
print(detect_langs("Dahil puno na sa restaurant, mom, shoti, and I were asked to share a seat with a customer. Buti nalang ikaw kasama namin sa table. Nice seeing you Shayane."))
print(detect_langs("asdfhu asdf;uih asdiufh asdui"))'''
from utility.NormalizeFilipino import NormalizeFilipino
from utility.PostCleaner import PostCleaner

'''import langid

from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier.set_languages(['tl', 'en'])

print(identifier.classify("Ahhhh tangina puso ko. Tama ka Joon Hyung my heart is fckn flattering"))'''

'''from PyDictionary import PyDictionary

dictionary = PyDictionary

print(dictionary.translate(dictionary, "bird", 'es'))'''

'''from polyglot.detect import Detector
text = "Dahil puno na sa restaurant, mom, shoti, and I were asked to share a seat with a customer. Buti nalang ikaw kasama namin sa table. Nice seeing you Shayane."
#detector = Detector(text)
#print(detector.language)

#for language in Detector(text).languages:
#    print(language)

from polyglot.text import Text
blob = "Ang pangalan ko ay Avril."
text = Text(blob)

for x in text.transliterate("en"):
    print(x)'''

'''import jpype
from jpype import *

jvmPath = jpype.getDefaultJVMPath()
jpype.startJVM(jvmPath, "-Djava.class.path=C:/Users/Avril_PC/Documents/Jars/NormAPI.jar")
normapi = JPackage("normapi").NormAPI
output = normapi.normalize_Text("and2 n q nkta m b q")
print(output)
jpype.shutdownJVM()
'''


post_cleaner = PostCleaner()

text = 'shiz life 💪 💪 💪💪💪'
print(post_cleaner.removeEmojis(text))