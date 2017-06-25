import pymysql
import pandas as pd
class DOM:
    def getConnection(self):
        return pymysql.connect(host='localhost', user='root', password='1234', db='twitterdb', charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    def getTwitterConnection(self):
        return pymysql.connect(host='localhost', user='root', password='1234', db='twitterremerged', charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    def getFacebookConnection(self):
        return pymysql.connect(host='localhost', user='root', password='1234', db='facebookdb', charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    def getData(self, conn):
        cursor = conn.cursor()

        sql = "SELECT P.Id, P.User, P.Text, hour(P.Time) as Time, P.CmbPOS, U.Age AS Age, U.Gender, U.Batch  FROM post P, user U WHERE P.User = U.Id"
        # sql = "select post.User, post.Text, hour(post.Time) as Time, post.CmbPOS, user.Batch, user.Age, user.Gender from post, user where post.User = user.Id  and (select count(*) from post as f where f.User = post.User and f.Time >= post.Time) <=10;"

        cursor.execute(sql)
        rows = cursor.fetchall()
        data = {'Features': [[row['User'],row['Text'],row['Time'], row['CmbPOS'], row['Batch']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}

        return data

    def getUserData(self, conn):
        cursor = conn.cursor()

        sql = "SELECT Id, Age, Gender, Batch FROM user U Order by Id"
        cursor.execute(sql)
        rows = cursor.fetchall()
        data = {'Features': [[row['Id'], row['Batch']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}

        return data


    def getTwitterUserData(self):
        data = self.getUserData(self.getTwitterConnection())

        X = pd.DataFrame(data['Features'], columns=['User', 'Batch'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y

    def getFBData(self):
        data =  self.getData(self.getFacebookConnection())

        X = pd.DataFrame(data['Features'], columns=['User', 'Text', 'PostTime', 'POS', 'Batch'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y

    def getTwitterData(self):
        data = self.getData(self.getTwitterConnection())

        X = pd.DataFrame(data['Features'], columns=['User', 'Text', 'PostTime', 'POS', 'Batch'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y

    def getMergedData(self):
        tdata = self.getData(self.getTwitterConnection())
        fdata = self.getData(self.getFacebookConnection())
        X = pd.DataFrame(tdata['Features']+fdata['Features'], columns=['User', 'Text', 'PostTime', 'POS', 'Batch'])
        y=pd.DataFrame(tdata['Results']+fdata['Results'], columns=['Age', 'Gender'])

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