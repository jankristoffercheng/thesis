'''from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory

DetectorFactory.seed = 0

print(detect_langs("Maglalagay sana ako ng caption na malalim kaso di ako artist."))
print(detect_langs("Dahil puno na sa restaurant, mom, shoti, and I were asked to share a seat with a customer. Buti nalang ikaw kasama namin sa table. Nice seeing you Shayane."))
print(detect_langs("asdfhu asdf;uih asdiufh asdui"))'''

'''import langid

from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier.set_languages(['tl', 'en'])

print(identifier.classify("Ahhhh tangina puso ko. Tama ka Joon Hyung my heart is fckn flattering"))'''

from PyDictionary import PyDictionary

dictionary = PyDictionary

print(dictionary.translate(dictionary, "bird", 'es'))

