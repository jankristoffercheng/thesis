import pymysql

class Connection:
    def getConnection(self):
        return pymysql.connect(host='localhost', user='root', password='root', db='twitterremerged', charset='utf8mb4',
                           cursorclass=pymysql.cursors.DictCursor, autocommit=True)