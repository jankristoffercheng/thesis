from sklearn.pipeline import Pipeline

from chaoticfolderofrissa.pipelinewraps.CharacterWrap import CharacterWrap
from chaoticfolderofrissa.pipelinewraps.ContextualWrap import ContextualWrap
from chaoticfolderofrissa.pipelinewraps.EmojiWrap import EmojiWrap
from chaoticfolderofrissa.pipelinewraps.FunctionWrap import FunctionWrap
from chaoticfolderofrissa.pipelinewraps.ItemSelector import ItemSelector
from chaoticfolderofrissa.pipelinewraps.LinkWrap import LinkWrap
from chaoticfolderofrissa.pipelinewraps.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.pipelinewraps.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.pipelinewraps.StructureWrap import StructureWrap
from chaoticfolderofrissa.pipelinewraps.WordWrap import WordWrap
import pandas as pd

from features.TFIDF import TFIDF
from utility.DataCleaner import DataCleaner


word = {'label':'Wrd.','categories':("Sixltr",
            "WPS",
          "informal")}
character = {'label':'Chr.','categories':("AllPunc",
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
            "OtherP")}
soc = {'label':'Soc.','categories':("swear",
            "netspeak",
            "assent",
            "nonflu",
            "filler")}

class FeatureExtract:
    def __init__(self, source, mindf, maxdf ):
        self.source=source
        self.posSeqPipeline = Pipeline([
            ('get_top', POSSeqWrap())
        ])
        self.timePipeline = Pipeline([
            ('extract', ItemSelector('PostTime')),
            ('enrange', PostTimeWrap())
        ])
        self.wordPipeline = Pipeline([
            ('extract', ItemSelector('Text')),
            ('process', WordWrap())
        ])
        self.characterPipeline = Pipeline([
            ('extract', ItemSelector('Text')),
            ('process', CharacterWrap())
        ])
        self.structurePipeline = Pipeline([
            ('extract', ItemSelector('Text')),
            ('process', StructureWrap())
        ])
        self.linkPipeline = Pipeline([
            ('extract', ItemSelector('Text')),
            ('process', LinkWrap())
        ])

        self.socLinContextPipeline = Pipeline([
            ('extract', ItemSelector('Text')),
            ('contextual', ContextualWrap())
        ])

        self.socLinEmojiPipeline = Pipeline([
            ('extract', ItemSelector('Text')),
            ('emoji', EmojiWrap())
        ])

        self.socLinFunctionPipeline = Pipeline([
            ('extract', ItemSelector('Text')),
            ('function', FunctionWrap())
        ])

        self.tfidf = TFIDF(mindf, maxdf)

        # print(X['Text'])

    def clean(self, x):
        return DataCleaner().clean_data(x)

    def get_liwc(self):
        df = pd.read_csv("data/"+self.source+"/raw/LIWC.csv", encoding = "ISO-8859-1", index_col=0)
        columns = []
        columnlabels = []
        for type in [word, character, soc]:
            label = type['label']
            for cat in type['categories']:
                columnlabels+=[label+cat]
                columns.append(cat)

        df= df[columns]
        df.columns=columnlabels
        return df




    def fit_transform(self, X):
        print("Extracting POS")
        posFeatures = self.posSeqPipeline.fit_transform(X)
        print("Extracting time")
        timeFeatures = self.timePipeline.fit_transform(X)
        print("Extracting word")
        wordFeatures = self.wordPipeline.fit_transform(X)
        print("Extracting character")
        characterFeatures = self.characterPipeline.fit_transform(X)
        print("Extracting structure")
        structureFeatures = self.structurePipeline.fit_transform(X)
        print("Extracting socLin")
        socLinFeatures = pd.concat([self.socLinContextPipeline.fit_transform(X), self.socLinEmojiPipeline.fit_transform(X),
                                    self.socLinFunctionPipeline.fit_transform(X)], axis=1)
        data = X['Text'].apply(self.clean)
        freq = self.tfidf.get_training_TFIDF(data)
        freqData = pd.DataFrame(data=freq.todense(),
                                columns=["Frq." + freq for freq in self.tfidf.getFeatureNames()])

        # print("Extracting link")
        # linkFeatures = self.linkPipeline.fit_transform(X)
        return pd.concat([posFeatures, freqData, timeFeatures, wordFeatures, characterFeatures, structureFeatures, socLinFeatures], axis=1)
        # return socLinFeatures

    def transform(self, X):
        print("Extracting POS")
        posFeatures = self.posSeqPipeline.fit_transform(X)
        print("Extracting time")
        timeFeatures = self.timePipeline.transform(X)
        print("Extracting word")
        wordFeatures = self.wordPipeline.transform(X)
        print("Extracting character")
        characterFeatures = self.characterPipeline.transform(X)
        print("Extracting structure")
        structureFeatures = self.structurePipeline.transform(X)
        print("Extracting socLin")
        socLinFeatures = pd.concat([self.socLinContextPipeline.transform(X), #self.socLinEmojiPipeline.transform(X),
                                    self.socLinFunctionPipeline.transform(X)], axis=1)
        print("Extracting tfidf")
        data = X['Text'].apply(self.clean)
        freq = self.tfidf.get_testing_TFIDF(data)
        freqData = pd.DataFrame(data=freq.todense(),
                                columns=["Frq." + freq for freq in self.tfidf.getFeatureNames()])

        # print("Extracting link")
        # linkFeatures = self.linkPipeline.fit_transform(X)
        return pd.concat(
            [posFeatures, freqData, timeFeatures, wordFeatures, characterFeatures, structureFeatures, socLinFeatures], axis=1)

    # def get_tfidf(self):

