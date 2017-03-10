from sklearn import svm, linear_model, metrics

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier

from chaoticfolderofrissa.AgeRangeWrap import AgeRangeWrap
from chaoticfolderofrissa.ItemSelector import ItemSelector
from chaoticfolderofrissa.ModelWrap import ModelWrap
from chaoticfolderofrissa.POSSeqWrap import POSSeqWrap
from chaoticfolderofrissa.DataFrameWrap import DataFrameWrap
from chaoticfolderofrissa.PostTimeWrap import PostTimeWrap
from chaoticfolderofrissa.SelectionWrap import SelectionWrap
from chaoticfolderofrissa.VectorizeWrap import VectorizeWrap
from connection.Connection import Connection

def getData():
    conn = Connection().getConnection()
    cursor = conn.cursor()
    sql = "SELECT P.Text, hour(P.PostTime) as Time, P.POS, (DATE_FORMAT(CURDATE(), '%Y') - DATE_FORMAT(U.Birthdate, '%Y') - (DATE_FORMAT(CURDATE(), '00-%m-%d') < DATE_FORMAT(U.Birthdate, '00-%m-%d'))) AS Age, U.Gender  FROM post P, user U WHERE P.User = U.Id;"
    cursor.execute(sql)
    row = cursor.fetchone()
    features = []
    results = []
    while row is not None:
        features.append([row['Text'],row['Time'], row['POS']])
        results.append([row['Age'], row['Gender']])
        row = cursor.fetchone()
    return {'Features': features, 'Results': results}


class MasterController:
    def __init__(self, features, results):
        self.X=pd.DataFrame(features, columns=['Text', 'PostTime', 'POS'])
        self.y=pd.DataFrame(results, columns=['Age', 'Gender'])

        wrap = AgeRangeWrap()
        self.y['Age'] = wrap.fit_transform(self.y['Age'])


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

    def trainCombined(self, selection, model):
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
        df =  pd.DataFrame(cm.predict(self.X))
        print('cm', confusion_matrix(self.y['Gender'], df))
        print('acc', metrics.accuracy_score(self.y['Gender'],df))

    def trainParallel(self, ageselect, agemodel, genselect, genmodel):
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
        print('agecm', confusion_matrix(self.y['Age'], agedf))
        print('ageacc', metrics.accuracy_score(self.y['Age'], agedf))

data = getData()
ctr = MasterController(data['Features'],data['Results'])
#ctr.trainAgetoGenderStack(SelectKBest(chi2, k=1000), MultinomialNB(),SelectKBest(chi2, k=1000), MultinomialNB())
#ctr.trainGendertoAgeStack(SelectKBest(chi2, k=1000), MultinomialNB(),SelectKBest(chi2, k=1000), MultinomialNB())
#ctr.trainCombined(SelectKBest(chi2, k=1000), MultinomialNB())
ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), MultinomialNB(), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), MultinomialNB(), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
#
#
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), svm.SVC(kernel='linear'), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
#
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), MultinomialNB(), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(chi2, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), SelectKBest(mutual_info_classif, k=1000), linear_model.Ridge(alpha=1.0))
#
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), PCA(n_components=100), svm.SVC(kernel='linear'))
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), PCA(n_components=100), DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99))
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), PCA(n_components=100), MultinomialNB())
# ctr.trainParallel(PCA(n_components=100), linear_model.Ridge(alpha=1.0), PCA(n_components=100), linear_model.Ridge(alpha=1.0))
