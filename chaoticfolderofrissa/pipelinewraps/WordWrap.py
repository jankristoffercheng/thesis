import uuid
from re import findall
from sklearn.base import TransformerMixin

import pandas as pd
from features.POSSequencePattern import POSSequencePattern
from features.Receptiviti import Receptiviti
from features.WordCount import WordCount
from model.Document import Document
from utility.DataCleaner import DataCleaner

columnList = ('wordCount',
              'capitalWordCount',
              'wordLength',
              'repLetterWords',
              'RatioUniqueToTotal',
              'totalShortWords',
              'netAbbrev',
              'lolhmm',
              'hapaxLegoRatio',
              'hapaxDislogoRatio',
              'yules',
              'simpsons',
              'sichels',
              'honores',
              'entropy')

wordfreqlab = ["wordfreq_"+str(num) for num in range (1,21)]
general = ("sixLtr",
            "wps")

labels = ("sixLtr",
            "wps",
          "informal")
# categories = ("informal")

class WordWrap(TransformerMixin):
    def __init__(self):
        self.recept = Receptiviti()
        self.features = WordCount()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        result = pd.DataFrame(columns=['Wrd.'+item for item in columnList+tuple(wordfreqlab)], dtype='float')
        for index, row in X.iteritems():
            # print(row)
            row = DataCleaner().clean_data(row)
            data = []
            data.append(self.features.getTotalNumberOfWords(row))
            data.append(self.features.getNWordsBegCapital(row))
            data.append(self.features.getAveLengthWords(row))
            data.append(self.features.getNWordsWithRepLetters(row))
            data.append(self.features.getRatioOfUniqueWords(row))
            data.append(self.features.getRatioOfShortWords(row))
            data.append(self.features.getRatioOfNetAbbrev(row))
            data.append(self.features.getLolHmmCount(row))
            data.append(self.features.getRatioOfHapaxLegomena(row))
            data.append(self.features.getRatioOfHapaxDislegomena(row))
            data.append(self.features.getYulesK(row))
            data.append(self.features.getSimpsonsD(row))
            data.append(self.features.getSichelsS(row))
            data.append(self.features.getHonoresR(row))
            data.append(self.features.getEntropy(row))
            data+=self.features.getWordLengthFreqDist(row)

            # print("hello")
            # person_name = str(uuid.uuid4()).replace("-", "")
            # person_id = self.recept.create_person(person_name)
            # print(row)
            # self.recept.add_content(person_id, row)
            # profile = self.recept.get_profile(person_id)
            # liwc = profile["liwc_scores"]
            # self.recept.delete_person(person_id)

            # print("bye")

            # data += [liwc[key] for key in general]
            # data.append(liwc["categories"]["informal"])
            # print(data)
            result.loc[index] = data

        return result