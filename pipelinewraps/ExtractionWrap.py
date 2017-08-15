from pandas import DataFrame
from sklearn.base import TransformerMixin


class ExtractionWrap(TransformerMixin):

    """
    Performs feature extraction
    """
    def __init__(self, extraction, target=None):
        """
        :param extraction: extraction technique
        :param target: true classes of the data
        """
        self.extraction = extraction
        self.target=target


    def fit(self, X, *args, **kwargs):
        self.extraction = self.extraction.fit(X, self.target)

        return self

    def transform(self, X, y=None, **transform_params):
        data=self.extraction.transform(X)
        result = DataFrame(data=data)
        return result