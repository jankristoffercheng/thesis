from sklearn import linear_model
import numpy as np

class RidgeRegression:
    def train(self, dimensions, classes):
        clf = linear_model.Ridge(alpha=1.0)
        X = np.array(dimensions)
        clf.fit(X, classes)
        return clf

    def classify(self, dimensions, classes, test):
        clf = self.train(dimensions, classes)
        tests = []
        tests.append(test)
        results = ['M' if x == 0 else 'F' for x in clf.predict(tests)]
        return results

    def classifyPersist(self, clf, test):
        tests = []
        tests.append(test)
        print(clf.predict(tests))
        results = ['M' if x == 0 else 'F' for x in clf.predict(tests)]
        return results