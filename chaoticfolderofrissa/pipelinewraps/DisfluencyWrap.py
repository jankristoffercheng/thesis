import uuid
from re import findall
from sklearn.base import TransformerMixin

import pandas as pd
from features.POSSequencePattern import POSSequencePattern
from features.Receptiviti import Receptiviti
from features.WordCount import WordCount
from model.Document import Document
from utility.DataCleaner import DataCleaner


categories = ("swear",
            "netspeak",
            "assent",
            "nonflu",
            "filler")

class DisfluencyWrap(TransformerMixin):
    def __init__(self):
        self.recept = Receptiviti()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        result = pd.DataFrame(columns=['Soc.'+item for item in categories], dtype='float')
        for index, row in X.iteritems():
            print(row)
            row = DataCleaner().clean_data(row)
            data = []

            person_name = str(uuid.uuid4()).replace("-", "")
            person_id = self.recept.create_person(person_name)
            print(row)
            self.recept.add_content(person_id, row)
            profile = self.recept.get_profile(person_id)
            liwc = profile["liwc_scores"]
            self.recept.delete_person(person_id)

            data += [liwc["categories"][key] for key in categories]
            result.loc[index] = data

        return result