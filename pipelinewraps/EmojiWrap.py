import pandas as pd
from sklearn.base import TransformerMixin

from features.EmojisEmoticons import EmojisEmoticons


class EmojiWrap(TransformerMixin):
    """
    Processes all emoji features of the data.
    TransformerMixin gives it the standard fit and transform functions to transform the data
    """

    def __init__(self, target=None):
        self.emoji = EmojisEmoticons()

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X, y=None, **transform_params):

        data = self.emoji.getEmojiTFIDF(X)
        df = pd.DataFrame(data=data.todense(),
                                 columns=["Soc."+freq for freq in self.emoji.getLabels()])
        # print("1")
        #df=DataFrame(data=data.todense(), columns=list(X.columns.values))
        # print(df.shape)
        return df