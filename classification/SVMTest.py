import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

#plt.scatter(x,y)
#plt.show()

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

#Xa = [1, 2, 3, 4, 11, 12, 13, 14, 15]
#Ya = [0, 0, 0, 0, 1, 1, 1, 1, 1]
y = [0,1,0,1,0,1]
# supervised learning, assigning 0 to "low" pairs and 1 to "high" pairs

clf = svm.SVC(kernel='linear', C=1, gamma=1)

clf.fit(X,y)



w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()

A = ([0.58, 0.76], [10.58, 10.76])

print(clf.predict(A))
#print(clf.predict(10.58,10.76))