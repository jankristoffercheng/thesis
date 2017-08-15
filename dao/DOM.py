import pandas as pd
import pymysql


class DOM:
    """
    Retrieves data from the database
    """
    def getTwitterConnection(self):
        """
        :return: Connection to the twitter db
        """
        return pymysql.connect(host='localhost', user='root', password='1234', db='twitterremerged', charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    def getFacebookConnection(self):
        """
        :return: Connection to the facebook db
        """
        return pymysql.connect(host='localhost', user='root', password='1234', db='facebookdb', charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    def getData(self, conn):
        """
        :param conn: connection to be used
        :return: all features of the text and the corresponding age and gender
        """
        cursor = conn.cursor()

        sql = "SELECT P.Id, P.User, P.Text, hour(P.Time) as Time, P.CmbPOS, U.Age AS Age, U.Gender, U.Batch  FROM post P, user U WHERE P.User = U.Id"
        # sql = "select post.User, post.Text, hour(post.Time) as Time, post.CmbPOS, user.Batch, user.Age, user.Gender from post, user where post.User = user.Id  and (select count(*) from post as f where f.User = post.User and f.Time >= post.Time) <=10;"

        cursor.execute(sql)
        rows = cursor.fetchall()
        data = {'Features': [[row['User'],row['Text'],row['Time'], row['CmbPOS'], row['Batch']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}

        return data

    def getUserData(self, conn):
        """

        :param conn: connection to be used
        :return: the batch, age, and gender of the users
        """
        cursor = conn.cursor()

        sql = "SELECT Id, Age, Gender, Batch FROM user U Order by Id"
        cursor.execute(sql)
        rows = cursor.fetchall()
        data = {'Features': [[row['Id'], row['Batch']] for row in rows], 'Results': [[row['Age'], row['Gender']] for row in rows]}

        return data


    def getTwitterUserData(self):
        """
        :return: twitter users data
        """
        data = self.getUserData(self.getTwitterConnection())

        X = pd.DataFrame(data['Features'], columns=['User', 'Batch'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y


    def getFBUserData(self):
        """

        :return: facebook users data
        """
        data = self.getUserData(self.getFacebookConnection())

        X = pd.DataFrame(data['Features'], columns=['User', 'Batch'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y

    def getFBData(self):
        """

        :return: facebook posts data
        """
        data =  self.getData(self.getFacebookConnection())

        X = pd.DataFrame(data['Features'], columns=['User', 'Text', 'PostTime', 'POS', 'Batch'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y

    def getTwitterData(self):
        """

        :return: twitter tweets data
        """
        data = self.getData(self.getTwitterConnection())

        X = pd.DataFrame(data['Features'], columns=['User', 'Text', 'PostTime', 'POS', 'Batch'])
        y=pd.DataFrame(data['Results'], columns=['Age', 'Gender'])

        return X,y

    def getMergedData(self):
        """

        :return: facebook and twitter user data
        """
        tdata = self.getData(self.getTwitterConnection())
        fdata = self.getData(self.getFacebookConnection())
        X = pd.DataFrame(fdata['Features']+tdata['Features'], columns=['User', 'Text', 'PostTime', 'POS', 'Batch'])
        y=pd.DataFrame(fdata['Results']+tdata['Results'], columns=['Age', 'Gender'])

        return X,y

    def getMergedUsersData(self):
        """

        :return: facebook posts and twitter tweets data
        """
        tdata = self.getUserData(self.getTwitterConnection())
        fdata = self.getUserData(self.getFacebookConnection())
        X = pd.DataFrame(fdata['Features']+tdata['Features'], columns=['User', 'Batch'])
        y=pd.DataFrame(fdata['Results']+tdata['Results'], columns=['Age', 'Gender'])

        return X,y
