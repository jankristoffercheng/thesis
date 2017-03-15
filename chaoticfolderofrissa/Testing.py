from collections import Counter

from sklearn.metrics import confusion_matrix

from chaoticfolderofrissa.AgeRangeWrap import AgeRangeWrap
from connection.Connection import Connection
from sklearn.externals import joblib
import pandas as pd
from sklearn import metrics

def getData():
    conn = Connection().getConnection()
    cursor = conn.cursor()
    sql = "Select count(*) as usernum from user"
    cursor.execute(sql)
    usernum = int(cursor.fetchone()['usernum'])
    print(usernum)

    cursor.execute("Select Id from user limit "+ str(int(usernum*0.6)))
    trainusers = [row['Id'] for row in cursor.fetchall()]
    print(trainusers)

    cursor.execute("Select Id from user limit 10000 offset " + str(int(usernum * 0.6)))
    testusers = [row['Id'] for row in cursor.fetchall()]
    print(testusers)

    sql = "SELECT P.User, P.Text, hour(P.PostTime) as Time, P.POS, (DATE_FORMAT(CURDATE(), '%Y') - DATE_FORMAT(U.Birthdate, '%Y') - (DATE_FORMAT(CURDATE(), '00-%m-%d') < DATE_FORMAT(U.Birthdate, '00-%m-%d'))) AS Age, U.Gender  FROM post P, user U WHERE P.User = U.Id and U.Id in (" + ",".join(map(str, trainusers)) + ")"
    cursor.execute(sql)
    rows = cursor.fetchall()
    traindata = {'Features': [[row['User'],row['Text'],row['Time'], row['POS']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}


    sql = "SELECT P.User, P.Text, hour(P.PostTime) as Time, P.POS, (DATE_FORMAT(CURDATE(), '%Y') - DATE_FORMAT(U.Birthdate, '%Y') - (DATE_FORMAT(CURDATE(), '00-%m-%d') < DATE_FORMAT(U.Birthdate, '00-%m-%d'))) AS Age, U.Gender  FROM post P, user U WHERE P.User = U.Id and U.Id in (" + ",".join(map(str, testusers)) + ")"
    cursor.execute(sql)
    rows = cursor.fetchall()
    testdata = {'Features': [[row['User'],row['Text'], row['Time'], row['POS']] for row in rows],
                 'Results': [[row['Age'], row['Gender']] for row in rows]}

    return {'Training': traindata,'Testing': testdata}

def test(cm, X, y):
    results = pd.Series(cm.predict(X))
    print('cm', confusion_matrix(y, results))
    print('acc', metrics.accuracy_score(y, results))
    # print('prec', metrics.precision_score(y, results))
    # print('rec', metrics.recall_score(y, results))
    # print('fm', metrics.f1_score(y, results))

    useres = []
    trueres = []
    ind = 0
    i = 1
    while(i<len(X.index)):
        if(X.loc[i]['User']!=X.loc[ind]['User']):
            df = X.loc[ind:i-1]
            pred = cm.predict(df)
            counter = Counter(pred)
            useres.append(counter.most_common(1)[0][0])
            trueres.append(y.loc[ind])
            ind = i

        i+=1

    print('cm', confusion_matrix(trueres, useres))
    print('acc', metrics.accuracy_score(trueres, useres))
    # print('prec', metrics.precision_score(trueres, useres))
    # print('rec', metrics.recall_score(trueres, useres))
    # print('fm', metrics.f1_score(trueres, useres))

data = getData()
trainX=pd.DataFrame(data['Training']['Features'], columns=['User', 'Text', 'PostTime', 'POS'])
trainY=pd.DataFrame(data['Training']['Results'], columns=['Age', 'Gender'])
testX=pd.DataFrame(data['Testing']['Features'], columns=['User', 'Text', 'PostTime', 'POS'])
testY=pd.DataFrame(data['Testing']['Results'], columns=['Age', 'Gender'])
wrap = AgeRangeWrap()
trainY['Age'] = wrap.fit_transform(trainY['Age'])
testY['Age'] = wrap.fit_transform(testY['Age'])

clf = joblib.load('cm/parallel/chi2_MNB_Gen.pkl')
test(clf, trainX, trainY['Gender'])
