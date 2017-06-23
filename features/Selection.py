import pandas
import numpy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


class Selection:
    def featureSelectChi2(self, num_features, data, result):
        test = SelectFpr(chi2)
        fit = test.fit(data, result)
        # summarize scores
        # numpy.set_printoptions(precision=3)
        # print(fit.scores_)
        features = fit.transform(data)
        # summarize selected features
        return features, test.get_support()

    def featureSelectMutualClassif(self, num_features, data, result):
        test = SelectFpr(mutual_info_classif)
        fit = test.fit(data, result)
        # summarize scores
        # numpy.set_printoptions(precision=3)
        # print(fit.scores_)
        features = fit.transform(data)
        # summarize selected features
        return features, test.get_support()

    def featureSelectMutualRegress(self, num_features, data, result):
        test = SelectFpr(mutual_info_regression)
        fit = test.fit(data, result)
        # summarize scores
        # numpy.set_printoptions(precision=3)
        # print(fit.scores_)
        features = fit.transform(data)
        # summarize selected features
        return features, test.get_support()

    def featureExtractSVD(self, num_components, data, result):
        svd = TruncatedSVD(n_components=num_components)
        fit = svd.fit(data)
        # summarize components
        # print("Explained Variance: %s") % fit.explained_variance_ratio_
        # print(fit.components_)
        components = svd.transform(data)
        return components

    def featureExtractPCA(self, num_components, data, result):
        pca = PCA(n_components=num_components)
        fit = pca.fit(data)
        # summarize components
        # print("Explained Variance: %s") % fit.explained_variance_ratio_
        # print(fit.components_)
        components = pca.transform(data)
        return components