import pandas as pd
from sklearn.base import TransformerMixin

from features.CharacterFeatures import CharacterFeatures
from utility.DataCleaner import DataCleaner

columnList = ('charCount',
              'letterCount',
              'upLetterCount',
              'digCount',
              'whiteCount',
              'repAlphaCount',
              'punctFrequency',
              'repPunctCount')

categories = ("AllPunc",
            "Period",
            "Comma",
            "Colon",
            "SemiC",
            "QMark",
            "Exclam",
            "Dash",
            "Quote",
            "Apostro",
            "Parenth",
            "OtherP")
class CharacterWrap(TransformerMixin):
    """
    Processes all character features of the data.
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        result = pd.DataFrame(columns=['Chr.'+item for item in columnList], dtype='float')
        for index, row in X.iteritems():
            # print(row)
            row = DataCleaner().clean_data(row)
            data = []
            data.append(CharacterFeatures().getTotalNumberOfCharacters(row))
            data.append(CharacterFeatures().getTotalNumberOfLetters(row))
            data.append(CharacterFeatures().getTotalNumberOfUppercase(row))
            data.append(CharacterFeatures().getTotalNumberOfDigitalNumbers(row))
            data.append(CharacterFeatures().getNumberOfWhiteSpaces(row))
            data.append(CharacterFeatures().getNumberOfRepetitiveAlphaCharacters(row))
            data.append(CharacterFeatures().getNumberOfSpecialChars(row))
            data.append(CharacterFeatures().getNumberOfRepeatedPunctuationMarks(row))

            # person_name = str(uuid.uuid4()).replace("-", "")
            # person_id = self.recept.create_person(person_name)
            # print(row)
            # self.recept.add_content(person_id, row)
            # profile = self.recept.get_profile(person_id)
            # liwc = profile["liwc_scores"]
            # self.recept.delete_person(person_id)

            # data += [liwc["categories"][key] for key in categories]
            result.loc[index] = data

        return result