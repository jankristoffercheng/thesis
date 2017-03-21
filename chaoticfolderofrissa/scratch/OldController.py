import pandas as pd
from chaoticfolderofrissa.AgeRangeWrap import AgeRangeWrap
from chaoticfolderofrissa.ItemSelector import ItemSelector
from chaoticfolderofrissa.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.SelectionWrap import SelectionWrap
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion

from chaoticfolderofrissa.pipelinewraps.VectorizeWrap import VectorizeWrap
from connection.Connection import Connection





class MasterController:
    def __init__(self, features, results, tfeatures, tresults):
        self.X=pd.DataFrame(features, columns=['User','Text', 'PostTime', 'POS'])
        self.y=pd.DataFrame(results, columns=['Age', 'Gender'])
        self.tX=pd.DataFrame(tfeatures, columns=['User','Text', 'PostTime', 'POS'])
        self.ty=pd.DataFrame(tresults, columns=['Age', 'Gender'])

        print(self.X.shape)
        wrap = AgeRangeWrap()
        self.y['Age'] = wrap.fit_transform(self.y['Age'])
        self.ty['Age'] = wrap.fit_transform(self.ty['Age'])

    def getFeatures(self, select):

         posseq = Pipeline([
                        ('get_top', POSSeqWrap())
                  ])

         trial = posseq.fit_transform(self.X, self.y['Gender'])


         features = FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(select))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange', PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1
                }
         )
         # result = features.fit_transform(self.X, self.y['Gender'])

         # numpy.savetxt("feature_train.csv", result.todense(), delimiter=',')
         # coo = result.tocoo(copy=False)
         #
         # df = pd.DataFrame({'index': coo.row, 'col': coo.col, 'data': coo.data}
         #                   )[['index', 'col', 'data']].sort_values(['index', 'col']
         #                                                           ).reset_index(drop=True)
         #
         # print("train",df.shape)
         # print(df)
         # result = features.transform(self.tX)
         # print("test", result.shape)
         # print(result)


    def trainAgetoGenderStack(self, ageselect, agemodel, genselect, genmodel):
        agepipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(ageselect))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange', PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1
                }
            )),
            ('estimator', agemodel)
        ])

        agecm = agepipeline.fit(self.X, self.y['Age'])
        agedf =  pd.DataFrame(agecm.predict(self.X))
        print('agecm', confusion_matrix(self.y['Age'], agedf))
        print('ageacc', metrics.accuracy_score(self.y['Age'], agedf))

        stackdf = self.X.copy()
        stackdf['Age'] = agecm.predict(self.X)

        genpipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(genselect))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange', PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ])),
                    ('age', Pipeline([
                        ('extract', ItemSelector('Age')),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1,
                    'age': 1
                }
            )),
            ('estimator', genmodel)
        ])

        gencm = genpipeline.fit(stackdf, self.y['Gender'])
        gendf =  pd.DataFrame(gencm.predict(stackdf))
        print('gencm', confusion_matrix(self.y['Gender'], gendf))
        print('genacc', metrics.accuracy_score(self.y['Gender'], gendf))

    def trainGendertoAgeStack(self, ageselect, agemodel, genselect, genmodel):
        genpipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(genselect))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange', PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1
                }
            )),
            ('estimator', genmodel)
        ])

        gencm = genpipeline.fit(self.X, self.y['Gender'])
        gendf = pd.DataFrame(gencm.predict(self.X))
        print('agecm', confusion_matrix(self.y['Gender'], gendf))
        print('ageacc', metrics.accuracy_score(self.y['Gender'], gendf))

        stackdf = pd.DataFrame(self.X)
        stackdf['Gender'] = gencm.predict(self.X)

        agepipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(ageselect))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange', PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ])),
                    ('gen', Pipeline([
                        ('extract', ItemSelector('Gender')),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1,
                    'gen': 1
                }
            )),
            ('estimator', agemodel)
        ])

        agecm = agepipeline.fit(stackdf, self.y['Age'])
        agedf = pd.DataFrame(agecm.predict(stackdf))
        print('agecm', confusion_matrix(self.y['Age'], agedf))
        print('ageacc', metrics.accuracy_score(self.y['Age'], agedf))

    def trainCombined(self, selection, model, file):
        pipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(selection,self.y[['Gender', 'Age']].apply(lambda x: ''.join(str(x)), axis=1)))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange',PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1
                }
            )),
            ('estimator', model)
        ])

        cm = pipeline.fit(self.X, self.y['Gender'])
        s = joblib.dump(cm, "cm/combined/"+file+"_Gender.pkl")
        df =  pd.DataFrame(cm.predict(self.X))
        print('cm', confusion_matrix(self.y['Gender'], df))
        print('acc', metrics.accuracy_score(self.y['Gender'],df))

        cm = pipeline.fit(self.X, self.y['Age'])
        s = joblib.dump(cm, "cm/combined/"+file+"_Age.pkl")
        df =  pd.DataFrame(cm.predict(self.X))
        print('cm', confusion_matrix(self.y['Age'], df))
        print('acc', metrics.accuracy_score(self.y['Age'],df))

    def trainParallel(self, ageselect, agemodel, genselect, genmodel, file):
        genpipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(genselect))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange', PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1
                }
            )),
            ('estimator', genmodel)
        ])

        gencm = genpipeline.fit(self.X, self.y['Gender'])
        s = joblib.dump(gencm, "cm/parallel/"+file+"_Gen.pkl")
        gendf = pd.DataFrame(gencm.predict(self.X))
        print('agecm', confusion_matrix(self.y['Gender'], gendf))
        print('ageacc', metrics.accuracy_score(self.y['Gender'], gendf))

        agepipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list=[
                    ('possequence', Pipeline([
                        ('get_top', POSSeqWrap()),
                    ])),
                    ('frequency', Pipeline([
                        ('extract', ItemSelector('Text')),
                        ('count', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('select', SelectionWrap(ageselect))
                    ])),
                    ('time', Pipeline([
                        ('extract', ItemSelector('PostTime')),
                        ('enrange', PostTimeWrap()),
                        ('vectorize', VectorizeWrap())
                    ]))
                ],
                transformer_weights={
                    'possequence': 1,
                    'frequency': 1,
                    'time': 1
                }
            )),
            ('estimator', agemodel)
        ])

        agecm = agepipeline.fit(self.X, self.y['Age'])
        agedf = pd.DataFrame(agecm.predict(self.X))
        s = joblib.dump(agecm, "cm/parallel/"+file+"_Age.pkl")
        print('agecm', confusion_matrix(self.y['Age'], agedf))
        print('ageacc', metrics.accuracy_score(self.y['Age'], agedf))

# data = getData()
# ctr = MasterController(data['Training']['Features'],data['Training']['Results'],data['Testing']['Features'],data['Testing']['Results'])
# ctr.getFeatures(SelectKBest(chi2, k=1000))
