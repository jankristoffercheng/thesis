import pymysql
import pandas as pd
class DOM:
    def getConnection(self):
        return pymysql.connect(host='localhost', user='root', password='1234', db='thesisdb', charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    def getData(self):
        conn = self.getConnection()
        cursor = conn.cursor()

        sql = "SELECT P.User, P.Text, hour(P.PostTime) as Time, P.CmbPOS, (DATE_FORMAT(CURDATE(), '%Y') - DATE_FORMAT(U.Birthdate, '%Y') - (DATE_FORMAT(CURDATE(), '00-%m-%d') < DATE_FORMAT(U.Birthdate, '00-%m-%d'))) AS Age, U.Gender  FROM post P, user U WHERE P.User = U.Id"
        cursor.execute(sql)
        rows = cursor.fetchall()
        data = {'Features': [[row['User'],row['Text'],row['Time'], row['CmbPOS']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}

        X = pd.DataFrame(data['Features'], columns=['User', 'Text', 'PostTime', 'POS'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y

    def get6040Data(self):
        conn = self.getConnection()
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

        sql = "SELECT P.User, P.Text, hour(P.PostTime) as Time, P.CmbPOS, (DATE_FORMAT(CURDATE(), '%Y') - DATE_FORMAT(U.Birthdate, '%Y') - (DATE_FORMAT(CURDATE(), '00-%m-%d') < DATE_FORMAT(U.Birthdate, '00-%m-%d'))) AS Age, U.Gender  FROM post P, user U WHERE P.User = U.Id and U.Id in (" + ",".join(map(str, trainusers)) + ")"
        cursor.execute(sql)
        rows = cursor.fetchall()
        traindata = {'Features': [[row['User'],row['Text'],row['Time'], row['CmbPOS']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}


        sql = "SELECT P.User, P.Text, hour(P.PostTime) as Time, P.CmbPOS, (DATE_FORMAT(CURDATE(), '%Y') - DATE_FORMAT(U.Birthdate, '%Y') - (DATE_FORMAT(CURDATE(), '00-%m-%d') < DATE_FORMAT(U.Birthdate, '00-%m-%d'))) AS Age, U.Gender  FROM post P, user U WHERE P.User = U.Id and U.Id in (" + ",".join(map(str, testusers)) + ")"
        cursor.execute(sql)
        rows = cursor.fetchall()
        testdata = {'Features': [[row['User'],row['Text'], row['Time'], row['CmbPOS']] for row in rows],
                     'Results': [[row['Age'], row['Gender']] for row in rows]}

        return {'Training': traindata,'Testing': testdata}