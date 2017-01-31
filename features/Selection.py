import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA



def featureSelectChi2(num_features, data, result):
    test = SelectKBest(chi2, k=num_features)
    fit = test.fit(data, result)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(data)
    # summarize selected features
    return features

def featureSelectMutualClassif(num_features, data, result):
    test = SelectKBest(mutual_info_classif, k=num_features)
    fit = test.fit(data, result)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(data)
    # summarize selected features
    return features

def featureSelectMutualRegress(num_features, data, result):
    test = SelectKBest(mutual_info_regression, k=num_features)
    fit = test.fit(data, result)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(data)
    # summarize selected features
    return features

def featureExtractPCA(num_components, data, result):
    pca = PCA(n_components=num_components)
    fit = pca.fit(data)
    # summarize components
    print("Explained Variance: %s") % fit.explained_variance_ratio_
    print(fit.components_)
    components = pca.transform(data)
    return components