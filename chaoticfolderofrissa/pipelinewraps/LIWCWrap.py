import uuid
from re import findall
from sklearn.base import TransformerMixin

import pandas as pd
from features.POSSequencePattern import POSSequencePattern
from features.Receptiviti import Receptiviti
from model.Document import Document

general = ("wc",
            "sixLtr",
            "wps")

categories = ("informal",
            "swear",
            "netspeak",
            "assent",
            "nonflu",
            "filler",
            "AllPunc",
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

class LIWCWrap(TransformerMixin):
    def __init__(self):
        self.recept = Receptiviti()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):


        result = pd.DataFrame(columns=general+categories, dtype='float')
        for index, row in X.iteritems():

            person_name = str(uuid.uuid4()).replace("-", "")
            person_id = self.recept.create_person(person_name)
            print(row)
            self.recept.add_content(person_id, row)
            profile = self.recept.get_profile(person_id)
            liwc = profile["liwc_scores"]
            self.recept.delete_person(person_id)

            data = [liwc[key] for key in general]
            data += [liwc["categories"][key] for key in categories]
            result.loc[index] = data

        return result